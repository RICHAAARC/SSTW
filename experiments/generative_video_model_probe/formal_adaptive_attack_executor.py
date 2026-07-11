"""逐视频执行完整论文 non-runtime 与 adaptive attack 协议。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

from evaluation.attacks.adaptive_video_optimizer import (
    ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL,
    ADAPTIVE_SOURCE_CLUSTER_SELECTION_PROTOCOL,
    ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL,
    optimize_bounded_parameter_attack_for_video,
    optimize_model_vae_regeneration_attack_for_video,
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
    _flow_key_context,
    _generation_key,
    _invoke_pipeline_loader,
    _load_pipeline,
    _prompt_text_by_id,
    _run_attacked_video_replay_for_model,
    build_flow_evidence_payload,
)
from experiments.generative_video_model_probe.colab_runtime import (
    validate_generation_model_provenance,
)
from main.methods.state_space_watermark.formal_detector import (
    FLOW_STATE_POSTERIOR_SCORE_SOURCE,
)
from experiments.generative_video_model_probe.formal_method_variants import (
    apply_frozen_flow_detector,
    frozen_flow_detector_calibration_from_governed_artifact,
)
from runtime.core.digest import build_stable_digest
from runtime.core.progress import ProgressReporter
from main.methods.state_space_watermark.replay_inversion import (
    REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID,
    ReplayGaussianLikelihoodConfig,
)


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL = "formal_adaptive_attack_execution"
FORMAL_ADAPTIVE_ATTACK_EXECUTION_GRANULARITY = (
    "per_video_frozen_flow_detector_adaptive_execution"
)

ADAPTIVE_SEARCH_PROTOCOLS: dict[str, tuple[str, str]] = {
    "generative_recompression_or_regeneration_attack": (
        "minimize_detector_score",
        "model_vae_regeneration",
    ),
    "endpoint_preserving_path_perturbation_attack": (
        "minimize_path_with_fixed_endpoint",
        "endpoint_path_perturbation",
    ),
    "detector_probing_with_public_negatives": (
        "minimize_detector_score",
        "public_detector_probe",
    ),
    "watermark_removal_optimization_attack": (
        "minimize_detector_score",
        "watermark_removal",
    ),
    "adversarial_detector_evasion_attack": (
        "minimize_detector_score",
        "detector_evasion",
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
    if (
        config.get("adaptive_attack_source_cluster_selection_protocol")
        != ADAPTIVE_SOURCE_CLUSTER_SELECTION_PROTOCOL
    ):
        raise ValueError("正式 adaptive source cluster 选择协议发生漂移")
    query_budget = int(config["adaptive_attack_query_budget_per_video"])
    query_budget_checkpoints = tuple(
        int(value) for value in config["adaptive_attack_query_budget_checkpoints"]
    )
    if (
        query_budget_checkpoints != tuple(sorted(set(query_budget_checkpoints)))
        or not query_budget_checkpoints
        or query_budget_checkpoints[0] < 1
        or query_budget_checkpoints[-1] != query_budget
    ):
        raise ValueError("正式 adaptive query-budget checkpoints 配置无效")
    public_probe_budget = int(
        config["adaptive_attack_public_negative_probe_query_budget"]
    )
    if public_probe_budget < 3 or public_probe_budget > query_budget:
        raise ValueError("public-negative probe 查询预算必须位于 [3, 主攻击预算]")
    return {
        "config": config,
        "paper_result_level": str(config["paper_result_level"]),
        "target_fpr": float(config["target_fpr"]),
        "required_protocols": tuple(required_non_runtime_attack_protocols_from_config(config)),
        "query_budget": query_budget,
        "query_budget_checkpoints": query_budget_checkpoints,
        "public_negative_probe_query_budget": public_probe_budget,
        "minimum_source_video_cluster_count_per_protocol": int(
            config[
                "minimum_adaptive_attack_source_video_cluster_count_per_protocol"
            ]
        ),
        "minimum_spoof_source_video_cluster_count": int(
            config[
                "minimum_independent_negative_video_count_for_fpr_upper_bound"
            ]
        ),
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
        calibrations[model_id] = (
            frozen_flow_detector_calibration_from_governed_artifact(row)
        )
    return calibrations


def _replay_likelihood_configs_by_model(
    run_root: Path,
) -> dict[str, ReplayGaussianLikelihoodConfig]:
    """加载逐模型冻结 replay 噪声模型，禁止 adaptive 查询期重新估计。"""

    rows = _read_jsonl(
        run_root
        / "thresholds"
        / "replay_gaussian_likelihood_calibrations.jsonl"
    )
    if not rows:
        raise RuntimeError("缺少逐模型 replay 高斯噪声 calibration artifact")
    configs: dict[str, ReplayGaussianLikelihoodConfig] = {}
    for row in rows:
        model_id = str(row.get("generation_model_id") or "").strip()
        if not model_id or model_id in configs:
            raise RuntimeError("replay 噪声 calibration 的模型标识缺失或重复")
        if (
            row.get("replay_likelihood_model_id")
            != REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID
            or row.get("replay_likelihood_calibration_protocol")
            != "calibration_clean_video_null_residual_cluster_equal_mle"
            or row.get("replay_likelihood_calibration_source_split") != "calibration"
            or row.get("replay_likelihood_calibration_status")
            != "fitted_from_model_specific_calibration_clean_videos"
            or row.get("test_time_likelihood_update_blocked") is not True
            or int(row.get("replay_likelihood_calibration_cluster_count") or 0) < 2
            or row.get("replay_likelihood_calibration_grid_policy")
            != "single_preregistered_primary_grid_for_noise_fit"
            or not isinstance(
                row.get("replay_likelihood_calibration_step_counts"), list
            )
            or len(row["replay_likelihood_calibration_step_counts"]) != 1
            or int(row["replay_likelihood_calibration_step_counts"][0]) < 2
        ):
            raise RuntimeError(f"replay 噪声 calibration artifact 不满足冻结协议: {model_id}")
        configs[model_id] = ReplayGaussianLikelihoodConfig(
            relative_observation_noise_standard_deviation=float(
                row["replay_relative_observation_noise_standard_deviation"]
            ),
            minimum_observation_noise_variance=float(
                row["replay_minimum_observation_noise_variance"]
            ),
            likelihood_model_id=str(row["replay_likelihood_model_id"]),
            calibration_protocol=str(
                row["replay_likelihood_calibration_protocol"]
            ),
            calibration_cluster_count=int(
                row["replay_likelihood_calibration_cluster_count"]
            ),
        )
    return configs


def _one_source_per_video(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个独立生成视频只选择一个固定输入, 避免把 runtime attack 重复当新视频。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster_id = str(row.get("statistical_cluster_id") or "").strip()
        if not cluster_id:
            raise RuntimeError("adaptive attack 正式输入缺少 source-video statistical cluster")
        grouped[cluster_id].append(row)
    preference = {"h264_crf18_runtime": 0, "h264_crf23_runtime": 1, "platform_transcode_runtime": 2}
    return [
        min(group, key=lambda row: (preference.get(str(row.get("attack_name")), 100), str(row.get("attack_name"))))
        for _cluster, group in sorted(grouped.items())
    ]


def _generation_spoof_source_population(
    run_root: Path,
    clean_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从全部 held-out full-method 生成视频构造 copy/spoof donor 总体。

    高成本 runtime attack 可以按 profile 只抽取30/100/200个 source cluster,
    但 fixed-FPR copy/spoof 误接受上界需要30/300/3000个独立 recipient。这里直接
    使用已经真实生成的完整方法视频作为 donor, 不复用攻击分数也不合成代理记录。
    """

    clean_by_identity = {
        (
            str(row.get("generation_model_id") or ""),
            str(row.get("prompt_id") or ""),
            str(row.get("seed_id") or ""),
        ): row
        for row in clean_rows
    }
    generation_rows = _read_jsonl(
        run_root / "records" / "generation_records.jsonl"
    )
    sources: list[dict[str, Any]] = []
    for row in generation_rows:
        if not (
            row.get("generation_status") == "success"
            and row.get("sample_role") == "attacked_positive_source"
            and row.get("method_variant") == "sstw_full_method"
            and row.get("split") == "test"
            and row.get("cross_model_role")
            != "cross_model_validation_model"
            and row.get("video_path")
            and row.get("video_sha256")
        ):
            continue
        identity = (
            str(row.get("generation_model_id") or ""),
            str(row.get("prompt_id") or ""),
            str(row.get("seed_id") or ""),
        )
        clean_row = clean_by_identity.get(identity)
        if clean_row is None:
            raise RuntimeError(
                "copy/spoof donor 缺少同模型、prompt、seed 的 held-out clean recipient"
            )
        sources.append({
            **row,
            "attacked_video_path": row["video_path"],
            "attacked_video_sha256": row["video_sha256"],
            "source_video_sha256": row["video_sha256"],
            "statistical_cluster_id": clean_row["statistical_cluster_id"],
            "statistical_independent_unit": "source_video_prompt_seed",
            "adaptive_attack_spoof_donor_source": (
                "heldout_full_method_generation_video"
            ),
        })
    return _one_source_per_video(sources)


def _select_preregistered_source_clusters(
    rows: list[dict[str, Any]],
    *,
    selected_cluster_count: int,
) -> list[dict[str, Any]]:
    """在读取 detector 结果前按稳定摘要选择预注册独立视频子集。

    该选择仅依赖 source-video cluster 标识, 不读取攻击分数、标签或质量结果。
    因此 probe、pilot 与 full 可以共享完全相同的攻击机制, 同时只在正式独立
    视频数量上扩展, 避免把全部 FPR 负样本规模错误地等同于高成本 adaptive 规模。
    """

    required = int(selected_cluster_count)
    if required <= 0:
        raise ValueError("adaptive attack 独立 source-video cluster 数必须为正数")
    if required > len(rows):
        raise RuntimeError(
            "adaptive attack 独立视频不足: "
            f"required={required}, observed={len(rows)}"
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            build_stable_digest({
                "selection_protocol": ADAPTIVE_SOURCE_CLUSTER_SELECTION_PROTOCOL,
                "statistical_cluster_id": str(row["statistical_cluster_id"]),
            }),
            str(row["statistical_cluster_id"]),
        ),
    )
    selected = ranked[:required]
    if len({str(row["statistical_cluster_id"]) for row in selected}) != required:
        raise RuntimeError("adaptive attack source-video cluster 选择出现重复独立单位")
    return selected


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
    likelihood_config: ReplayGaussianLikelihoodConfig,
    *,
    prompt: str,
    key_text: str,
) -> Callable[[Path], Mapping[str, Any]]:
    """把真实视频映射为同一个冻结 Flow 后验, 查询期间不重新拟合。"""

    key_context = _flow_key_context(prompt, pipeline.scheduler)

    def score(video_path: Path) -> Mapping[str, Any]:
        replay = _run_attacked_video_replay_for_model(
            pipeline,
            video_path,
            prompt=prompt,
            key_text=key_text,
            key_context=key_context,
            likelihood_config=likelihood_config,
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
        "generation_model_family": source.get("generation_model_family"),
        "generation_model_requested_revision": source.get(
            "generation_model_requested_revision"
        ),
        "generation_model_commit_or_hash": source.get(
            "generation_model_commit_or_hash"
        ),
        "generation_model_revision_source": source.get(
            "generation_model_revision_source"
        ),
        "generation_model_revision_resolution_status": source.get(
            "generation_model_revision_resolution_status"
        ),
        "prompt_id": source.get("prompt_id"),
        "seed_id": source.get("seed_id"),
        "trajectory_trace_id": source.get("trajectory_trace_id"),
        "statistical_cluster_id": source.get("statistical_cluster_id"),
        "adaptive_attack_source_statistical_cluster_id": source.get(
            "statistical_cluster_id"
        ),
        "adaptive_attack_spoof_donor_source": source.get(
            "adaptive_attack_spoof_donor_source"
        ),
        "statistical_independent_unit": "source_video_prompt_seed",
        "split": source.get("split"),
        "method_variant": "sstw_full_method",
        "adaptive_attack_status": "ready",
        "metric_status": "measured_formal",
        "adaptive_attack_evidence_level": FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        "adaptive_attack_execution_granularity": (
            FORMAL_ADAPTIVE_ATTACK_EXECUTION_GRANULARITY
        ),
        "adaptive_attack_score_orientation": "higher_is_more_watermarked",
        "adaptive_attack_query_budget": context["query_budget"],
        "adaptive_attack_query_budget_checkpoints": list(
            context["query_budget_checkpoints"]
        ),
        "adaptive_attack_query_budget_checkpoint_protocol": (
            ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
        ),
        "adaptive_attack_source_cluster_selection_protocol": (
            ADAPTIVE_SOURCE_CLUSTER_SELECTION_PROTOCOL
        ),
        "minimum_adaptive_attack_source_video_cluster_count_per_protocol": (
            context["minimum_source_video_cluster_count_per_protocol"]
        ),
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


def audit_adaptive_optimizer_evidence(
    records: list[Mapping[str, Any]],
    *,
    query_budget: int,
    query_budget_checkpoints: tuple[int, ...] | None = None,
    public_negative_probe_query_budget: int | None = None,
) -> dict[str, bool]:
    """核验 detector-feedback 搜索、public probe 与模型 VAE 重生成均为真实执行。"""

    expected_checkpoints = tuple(
        int(value) for value in (query_budget_checkpoints or (query_budget,))
    )
    if public_negative_probe_query_budget is None:
        expected_public_probe_budget = int(query_budget)
        expected_public_checkpoints = expected_checkpoints
    else:
        expected_public_probe_budget = int(public_negative_probe_query_budget)
        expected_public_checkpoints = (1, expected_public_probe_budget)

    search_rows = [
        row
        for row in records
        if row.get("non_runtime_attack_protocol") in ADAPTIVE_SEARCH_PROTOCOLS
    ]

    def checkpoint_evidence_ready(
        row: Mapping[str, Any],
        *,
        candidates_field: str,
        checkpoints_field: str,
        checkpoint_records_field: str,
        objective: str,
        checkpoints: tuple[int, ...] = expected_checkpoints,
    ) -> bool:
        """核验每个 checkpoint 只选择其真实查询前缀中的最优可接受候选。"""

        candidates = row.get(candidates_field)
        checkpoint_records = row.get(checkpoint_records_field)
        if (
            list(row.get(checkpoints_field) or []) != list(checkpoints)
            or not isinstance(candidates, list)
            or not isinstance(checkpoint_records, list)
            or len(checkpoint_records) != len(checkpoints)
        ):
            return False

        for checkpoint, checkpoint_record in zip(
            checkpoints,
            checkpoint_records,
        ):
            prefix = candidates[:checkpoint]
            feasible = [
                candidate for candidate in prefix
                if candidate.get("admissible") is True
            ]
            if not feasible:
                return False
            if objective == "minimize_path_with_fixed_endpoint":
                selected = min(
                    feasible,
                    key=lambda candidate: (
                        float(candidate["path_score"]),
                        float(candidate["detector_score"]),
                        int(candidate["candidate_index"]),
                    ),
                )
            else:
                selected = min(
                    feasible,
                    key=lambda candidate: (
                        float(candidate["detector_score"]),
                        float(candidate["path_score"]),
                        int(candidate["candidate_index"]),
                    ),
                )
            if not (
                checkpoint_record.get("adaptive_attack_checkpoint_selection_protocol")
                == ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
                and int(
                    checkpoint_record.get(
                        "adaptive_attack_query_budget_checkpoint"
                    )
                    or 0
                )
                == checkpoint
                and int(
                    checkpoint_record.get(
                        "adaptive_attack_checkpoint_observed_query_count"
                    )
                    or 0
                )
                == checkpoint
                and int(
                    checkpoint_record.get(
                        "adaptive_attack_checkpoint_candidate_count"
                    )
                    or 0
                )
                == checkpoint
                and int(
                    checkpoint_record.get(
                        "adaptive_attack_checkpoint_admissible_candidate_count"
                    )
                    or 0
                )
                == len(feasible)
                and checkpoint_record.get(
                    "adaptive_attack_checkpoint_has_admissible_candidate"
                )
                is True
                and int(
                    checkpoint_record.get(
                        "adaptive_attack_checkpoint_selected_candidate_index"
                    )
                    or 0
                )
                == int(selected["candidate_index"])
                and checkpoint_record.get(
                    "adaptive_attack_checkpoint_output_video_sha256"
                )
                == selected.get("video_sha256")
                and float(
                    checkpoint_record["adaptive_attack_checkpoint_detector_score"]
                )
                == float(selected["detector_score"])
                and checkpoint_record.get(
                    "adaptive_attack_checkpoint_detected_by_sstw"
                )
                == bool(selected["decision"])
            ):
                return False
        return True

    def common_search_evidence_ready(row: Mapping[str, Any]) -> bool:
        candidates = row.get("adaptive_attack_candidate_records")
        public_query_count = int(
            row.get("adaptive_attack_public_negative_probe_count") or 0
        )
        return bool(
            int(row.get("adaptive_attack_query_count") or 0) == int(query_budget)
            and isinstance(row.get("adaptive_attack_selected_parameters"), Mapping)
            and isinstance(candidates, list)
            and len(candidates) == int(query_budget)
            and len({
                round(
                    float(candidate["attack_parameters"]["attack_strength"]),
                    12,
                )
                for candidate in candidates
                if isinstance(candidate.get("attack_parameters"), Mapping)
                and candidate["attack_parameters"].get("attack_strength") is not None
            })
            == int(query_budget)
            and row.get("adaptive_attack_query_budget_checkpoint_protocol")
            == ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
            and row.get("adaptive_attack_query_accounting_protocol")
            == "all_target_and_public_negative_frozen_detector_calls"
            and int(
                row.get("adaptive_attack_total_detector_query_count") or 0
            )
            == int(query_budget) + public_query_count
            and checkpoint_evidence_ready(
                row,
                candidates_field="adaptive_attack_candidate_records",
                checkpoints_field="adaptive_attack_query_budget_checkpoints",
                checkpoint_records_field=(
                    "adaptive_attack_query_budget_checkpoint_records"
                ),
                objective=str(row.get("adaptive_attack_objective") or ""),
            )
        )

    def two_coordinate_search_evidence_ready(row: Mapping[str, Any]) -> bool:
        candidates = row.get("adaptive_attack_candidate_records")
        if not isinstance(candidates, list):
            return False
        coordinate_pairs = {
            (
                round(float(candidate["adaptive_search_coordinate_1_value"]), 12),
                round(float(candidate["adaptive_search_coordinate_2_value"]), 12),
            )
            for candidate in candidates
            if candidate.get("adaptive_search_coordinate_1_value") is not None
            and candidate.get("adaptive_search_coordinate_2_value") is not None
        }
        expected_initial_phases = (
            "base_point",
            "coordinate_1_probe",
            "coordinate_2_probe",
        )
        initial_phases_ready = tuple(
            candidate.get("adaptive_search_query_phase")
            for candidate in candidates[:3]
        ) == expected_initial_phases
        feedback_rows = candidates[3:]
        feedback_ready = all(
            candidate.get("adaptive_search_query_phase")
            == "detector_feedback_pattern_refinement"
            and candidate.get("adaptive_search_feedback_parent_candidate_index")
            is not None
            for candidate in feedback_rows
        )
        return bool(
            row.get("adaptive_attack_optimizer_type")
            == "sequential_detector_feedback_two_coordinate_pattern_search"
            and row.get("adaptive_search_protocol")
            == ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL
            and len(coordinate_pairs) == int(query_budget)
            and initial_phases_ready
            and feedback_ready
        )

    search_ready = bool(search_rows) and all(
        common_search_evidence_ready(row)
        and (
            row.get("adaptive_attack_optimizer_type")
            == "sequential_detector_feedback_one_coordinate_model_vae_search"
            if row.get("non_runtime_attack_protocol")
            == "generative_recompression_or_regeneration_attack"
            else two_coordinate_search_evidence_ready(row)
        )
        for row in search_rows
    )
    regeneration_rows = [
        row
        for row in search_rows
        if row.get("non_runtime_attack_protocol")
        == "generative_recompression_or_regeneration_attack"
    ]
    regeneration_ready = bool(regeneration_rows) and all(
        all(
            isinstance(candidate.get("attack_parameters"), Mapping)
            and candidate["attack_parameters"].get("model_vae_regeneration_status")
            == "measured_model_vae_encode_perturb_decode"
            and bool(candidate["attack_parameters"].get("model_vae_class"))
            and candidate["attack_parameters"].get(
                "model_vae_noise_direction_policy"
            )
            == "fixed_per_source_video_across_strength_queries"
            and int(
                candidate["attack_parameters"].get("model_vae_source_frame_count")
                or 0
            )
            > 0
            and int(
                candidate["attack_parameters"].get("model_vae_output_frame_count")
                or 0
            )
            > 0
            for candidate in row["adaptive_attack_candidate_records"]
        )
        for row in regeneration_rows
    )
    public_rows = [
        row
        for row in search_rows
        if row.get("non_runtime_attack_protocol")
        == "detector_probing_with_public_negatives"
    ]
    public_probe_ready = bool(public_rows) and all(
        int(row.get("adaptive_attack_public_negative_probe_count") or 0)
        == expected_public_probe_budget
        and isinstance(
            row.get("adaptive_attack_public_negative_candidate_records"), list
        )
        and len(row["adaptive_attack_public_negative_candidate_records"])
        == expected_public_probe_budget
        and row.get("adaptive_attack_public_negative_informed_strength") is not None
        and checkpoint_evidence_ready(
            row,
            candidates_field="adaptive_attack_public_negative_candidate_records",
            checkpoints_field=(
                "adaptive_attack_public_negative_query_budget_checkpoints"
            ),
            checkpoint_records_field=(
                "adaptive_attack_public_negative_query_budget_checkpoint_records"
            ),
            objective="minimize_detector_score",
            checkpoints=expected_public_checkpoints,
        )
        for row in public_rows
    )
    return {
        "adaptive_detector_feedback_search_ready": search_ready,
        "adaptive_model_vae_regeneration_ready": regeneration_ready,
        "adaptive_public_negative_probe_ready": public_probe_ready,
        "adaptive_query_budget_checkpoint_ready": bool(search_rows) and all(
            checkpoint_evidence_ready(
                row,
                candidates_field="adaptive_attack_candidate_records",
                checkpoints_field="adaptive_attack_query_budget_checkpoints",
                checkpoint_records_field=(
                    "adaptive_attack_query_budget_checkpoint_records"
                ),
                objective=str(row.get("adaptive_attack_objective") or ""),
            )
            for row in search_rows
        ),
    }


def build_adaptive_query_budget_checkpoint_records(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """把单视频嵌套 checkpoint 展开为可独立重建论文曲线的 governed records。"""

    checkpoint_records: list[dict[str, Any]] = []
    for row in records:
        protocol = str(row.get("non_runtime_attack_protocol") or "")
        if protocol not in ADAPTIVE_SEARCH_PROTOCOLS:
            continue
        source_cluster_id = str(
            row.get("adaptive_attack_source_statistical_cluster_id") or ""
        )
        for checkpoint in row.get(
            "adaptive_attack_query_budget_checkpoint_records",
            [],
        ):
            payload = {
                "record_version": "formal_adaptive_query_budget_checkpoint_v1",
                "paper_result_level": row.get("paper_result_level"),
                "target_fpr": row.get("target_fpr"),
                "non_runtime_attack_protocol": protocol,
                "generation_model_id": row.get("generation_model_id"),
                "prompt_id": row.get("prompt_id"),
                "seed_id": row.get("seed_id"),
                "trajectory_trace_id": row.get("trajectory_trace_id"),
                "statistical_cluster_id": source_cluster_id,
                "adaptive_attack_source_statistical_cluster_id": source_cluster_id,
                "statistical_independent_unit": "source_video_prompt_seed",
                "adaptive_attack_checkpoint_query_role": "heldout_test_video",
                "adaptive_attack_objective": row.get("adaptive_attack_objective"),
                "adaptive_attack_query_budget": row.get(
                    "adaptive_attack_query_budget"
                ),
                "adaptive_attack_query_budget_checkpoints": row.get(
                    "adaptive_attack_query_budget_checkpoints"
                ),
                "adaptive_attack_query_budget_checkpoint_protocol": row.get(
                    "adaptive_attack_query_budget_checkpoint_protocol"
                ),
                "adaptive_attack_checkpoint_public_negative_query_count": int(
                    row.get("adaptive_attack_public_negative_probe_count") or 0
                ),
                "adaptive_attack_checkpoint_total_detector_query_count": (
                    int(
                        checkpoint.get(
                            "adaptive_attack_query_budget_checkpoint"
                        )
                        or 0
                    )
                    + int(
                        row.get("adaptive_attack_public_negative_probe_count")
                        or 0
                    )
                ),
                "test_time_threshold_update_blocked": row.get(
                    "test_time_threshold_update_blocked"
                ),
                **dict(checkpoint),
            }
            checkpoint_records.append({
                "formal_adaptive_query_budget_checkpoint_record_id": (
                    "adaptive_checkpoint_"
                    f"{build_stable_digest(payload)[:16]}"
                ),
                **payload,
            })
    return checkpoint_records


def audit_adaptive_source_cluster_coverage(
    records: list[Mapping[str, Any]],
    checkpoint_records: list[Mapping[str, Any]],
    *,
    required_protocols: tuple[str, ...],
    minimum_source_cluster_count: int,
    minimum_spoof_source_cluster_count: int | None = None,
    query_budget_checkpoints: tuple[int, ...],
    expected_retention_source_cluster_ids: set[str] | None = None,
    expected_spoof_source_cluster_ids: set[str] | None = None,
) -> dict[str, Any]:
    """按预注册独立单位核验协议覆盖, 并显式阻断伪重复。"""

    minimum_count = int(minimum_source_cluster_count)
    spoof_minimum = int(minimum_spoof_source_cluster_count or minimum_count)
    expected_source_clusters = set(expected_retention_source_cluster_ids or ())
    if not expected_source_clusters:
        expected_source_clusters = {
            str(row.get("adaptive_attack_source_statistical_cluster_id") or "")
            for row in records
            if row.get("non_runtime_attack_protocol")
            not in {
                "watermark_spoofing_or_copy_attack",
                "collusion_multi_sample_attack",
            }
            and row.get("adaptive_attack_source_statistical_cluster_id")
        }
    expected_spoof_clusters = set(expected_spoof_source_cluster_ids or ())
    if not expected_spoof_clusters:
        expected_spoof_clusters = {
            str(row.get("adaptive_attack_source_statistical_cluster_id") or "")
            for row in records
            if row.get("non_runtime_attack_protocol")
            == "watermark_spoofing_or_copy_attack"
            and row.get("adaptive_attack_source_statistical_cluster_id")
        } or set(expected_source_clusters)
    protocol_cluster_counts: dict[str, int] = {}
    duplicate_independent_units: list[str] = []
    incomplete_protocols: list[str] = []

    for protocol in required_protocols:
        protocol_rows = [
            row for row in records
            if row.get("non_runtime_attack_protocol") == protocol
        ]
        if protocol == "collusion_multi_sample_attack":
            pair_ids = [
                str(row.get("statistical_cluster_id") or "")
                for row in protocol_rows
            ]
            member_ids = [
                str(member)
                for row in protocol_rows
                for member in row.get(
                    "adaptive_attack_member_statistical_cluster_ids",
                    [],
                )
            ]
            protocol_cluster_counts[protocol] = len(set(pair_ids))
            duplicate_independent_units.extend(
                f"{protocol}::{pair_id}"
                for pair_id in sorted(set(pair_ids))
                if pair_ids.count(pair_id) != 1
            )
            collusion_ready = (
                len(protocol_rows) == minimum_count // 2
                and len(pair_ids) == len(set(pair_ids))
                and len(member_ids) == minimum_count
                and len(member_ids) == len(set(member_ids))
                and set(member_ids) == expected_source_clusters
                and all(
                    row.get("statistical_independent_unit")
                    == "disjoint_source_video_pair"
                    for row in protocol_rows
                )
            )
            if not collusion_ready:
                incomplete_protocols.append(protocol)
            continue

        source_ids = [
            str(row.get("adaptive_attack_source_statistical_cluster_id") or "")
            for row in protocol_rows
        ]
        protocol_cluster_counts[protocol] = len(set(source_ids))
        duplicate_independent_units.extend(
            f"{protocol}::{source_id}"
            for source_id in sorted(set(source_ids))
            if source_ids.count(source_id) != 1
        )
        expected_protocol_clusters = (
            expected_spoof_clusters
            if protocol == "watermark_spoofing_or_copy_attack"
            else expected_source_clusters
        )
        expected_protocol_count = (
            spoof_minimum
            if protocol == "watermark_spoofing_or_copy_attack"
            else minimum_count
        )
        if not (
            len(protocol_rows) == expected_protocol_count
            and len(source_ids) == len(set(source_ids))
            and set(source_ids) == expected_protocol_clusters
        ):
            incomplete_protocols.append(protocol)

    missing_checkpoint_scopes: list[str] = []
    for protocol in ADAPTIVE_SEARCH_PROTOCOLS:
        if protocol not in required_protocols:
            continue
        for checkpoint in query_budget_checkpoints:
            scoped = [
                row for row in checkpoint_records
                if row.get("non_runtime_attack_protocol") == protocol
                and int(
                    row.get("adaptive_attack_query_budget_checkpoint") or 0
                )
                == int(checkpoint)
            ]
            cluster_ids = [
                str(row.get("statistical_cluster_id") or "") for row in scoped
            ]
            if not (
                len(scoped) == minimum_count
                and len(cluster_ids) == len(set(cluster_ids))
                and set(cluster_ids) == expected_source_clusters
                and all(
                    row.get("adaptive_attack_checkpoint_has_admissible_candidate")
                    is True
                    and row.get("adaptive_attack_checkpoint_output_video_sha256")
                    for row in scoped
                )
            ):
                missing_checkpoint_scopes.append(f"{protocol}::q={checkpoint}")

    coverage_ready = (
        len(expected_source_clusters) == minimum_count
        and len(expected_spoof_clusters) == spoof_minimum
        and expected_source_clusters <= expected_spoof_clusters
        and not incomplete_protocols
    )
    uniqueness_ready = not duplicate_independent_units
    checkpoint_ready = not missing_checkpoint_scopes
    return {
        "adaptive_attack_source_cluster_selection_protocol": (
            ADAPTIVE_SOURCE_CLUSTER_SELECTION_PROTOCOL
        ),
        "minimum_adaptive_attack_source_video_cluster_count_per_protocol": (
            minimum_count
        ),
        "minimum_adaptive_spoof_source_video_cluster_count": spoof_minimum,
        "adaptive_attack_selected_source_video_cluster_count": len(
            expected_source_clusters
        ),
        "adaptive_attack_spoof_source_video_cluster_count": len(
            expected_spoof_clusters
        ),
        "adaptive_attack_protocol_independent_cluster_counts": (
            protocol_cluster_counts
        ),
        "adaptive_attack_incomplete_cluster_protocols": sorted(
            incomplete_protocols
        ),
        "adaptive_attack_duplicate_independent_units": sorted(
            duplicate_independent_units
        ),
        "adaptive_attack_missing_query_budget_checkpoint_scopes": sorted(
            missing_checkpoint_scopes
        ),
        "adaptive_attack_source_cluster_coverage_decision": (
            "PASS" if coverage_ready else "FAIL"
        ),
        "adaptive_attack_independent_unit_uniqueness_decision": (
            "PASS" if uniqueness_ready else "FAIL"
        ),
        "adaptive_attack_query_budget_checkpoint_coverage_decision": (
            "PASS" if checkpoint_ready else "FAIL"
        ),
        "adaptive_attack_source_cluster_coverage_ready": coverage_ready,
        "adaptive_attack_independent_unit_uniqueness_ready": uniqueness_ready,
        "adaptive_attack_query_budget_checkpoint_coverage_ready": checkpoint_ready,
    }


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
    replay_likelihood_configs = _replay_likelihood_configs_by_model(root)
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
    source_population = _one_source_per_video(positives)
    if not source_population:
        raise RuntimeError("缺少 held-out test full-method 视频, 无法执行 adaptive attack")
    retention_sources = _select_preregistered_source_clusters(
        source_population,
        selected_cluster_count=(
            context["minimum_source_video_cluster_count_per_protocol"]
        ),
    )
    spoof_source_population = _generation_spoof_source_population(root, clean)
    spoof_sources = _select_preregistered_source_clusters(
        spoof_source_population,
        selected_cluster_count=(
            context["minimum_spoof_source_video_cluster_count"]
        ),
    )
    retention_cluster_ids = {
        str(row["statistical_cluster_id"]) for row in retention_sources
    }
    spoof_cluster_ids = {
        str(row["statistical_cluster_id"]) for row in spoof_sources
    }
    if not retention_cluster_ids <= spoof_cluster_ids:
        raise RuntimeError("adaptive retention 子集必须嵌套于 spoof 固定 FPR 样本")
    sources = [*retention_sources, *spoof_sources]
    if (
        "collusion_multi_sample_attack" in context["required_protocols"]
        and (len(retention_sources) < 4 or len(retention_sources) % 2 != 0)
    ):
        _disjoint_collusion_peer_index(0, len(retention_sources))
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
    missing_replay_likelihood_models = sorted(
        set(models) - set(replay_likelihood_configs)
    )
    if missing_replay_likelihood_models:
        raise RuntimeError(
            "adaptive attack 缺少模型专属冻结 replay 噪声模型: "
            f"{missing_replay_likelihood_models}"
        )
    model_revisions: dict[str, str] = {}
    for source in sources:
        model_id = str(source["generation_model_id"])
        revision = validate_generation_model_provenance(source)
        previous = model_revisions.setdefault(model_id, revision)
        if previous != revision:
            raise RuntimeError(f"adaptive attack 同一模型混用了多个 revision: {model_id}")
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=model_revisions[model_id],
        )
        for model_id in models
    }
    records: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    collusion_required = (
        "collusion_multi_sample_attack" in context["required_protocols"]
    )
    spoof_required = (
        "watermark_spoofing_or_copy_attack" in context["required_protocols"]
    )
    expected_count = (
        len(retention_sources)
        * (len(context["required_protocols"]) - int(spoof_required))
        - (len(retention_sources) // 2 if collusion_required else 0)
        + (len(spoof_sources) if spoof_required else 0)
    )
    progress = ProgressReporter(
        "formal_per_video_adaptive_attack",
        expected_count,
        "video_protocol",
    )
    progress_index = 0
    non_spoof_protocols = tuple(
        protocol
        for protocol in context["required_protocols"]
        if protocol != "watermark_spoofing_or_copy_attack"
    )
    execution_source_groups = [
        (source_index, source, non_spoof_protocols)
        for source_index, source in enumerate(retention_sources)
    ] + [
        (
            source_index,
            source,
            ("watermark_spoofing_or_copy_attack",),
        )
        for source_index, source in enumerate(spoof_sources)
        if spoof_required
    ]
    for source_index, source, source_protocols in execution_source_groups:
        model_id = str(source["generation_model_id"])
        prompt = prompt_map[str(source["prompt_id"])]
        key_text = _generation_key(source)
        pipeline = pipelines[model_id]
        scorer = _build_scorer(
            pipeline,
            calibrations[model_id],
            replay_likelihood_configs[model_id],
            prompt=prompt,
            key_text=key_text,
        )
        source_video = _resolve_video(root, source.get("attacked_video_path"), "attacked_videos")
        for protocol in source_protocols:
            # collusion 的统计单位是预注册的不重叠视频对。每对只执行并记录一次,
            # 避免把同一对按两个成员方向重复生成后伪装成两个独立样本。
            if protocol == "collusion_multi_sample_attack" and source_index % 2 == 1:
                continue
            progress_index += 1
            progress.update(progress_index, f"video={source_index} protocol={protocol}")
            base = _base_record(protocol, source, context)
            base["adaptive_attack_input_video_path"] = str(source_video)
            base["adaptive_attack_input_video_sha256"] = source.get("attacked_video_sha256")
            try:
                if protocol in ADAPTIVE_SEARCH_PROTOCOLS:
                    objective, attack_family = ADAPTIVE_SEARCH_PROTOCOLS[protocol]
                    public_probe_summary: dict[str, Any] = {}
                    public_informed_strength: float | None = None
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
                            public_revision = validate_generation_model_provenance(
                                public_row
                            )
                            pipelines[public_model_id] = _invoke_pipeline_loader(
                                pipeline_loader,
                                model_id=public_model_id,
                                revision=public_revision,
                            )
                        if public_model_id not in replay_likelihood_configs:
                            raise RuntimeError(
                                "public negative 缺少模型专属冻结 replay 噪声模型"
                            )
                        public_prompt = prompt_map[str(public_row["prompt_id"])]
                        public_trial_index = int(public_row.get("clean_negative_trial_index") or 0)
                        public_key = _generation_key(
                            public_row,
                            extra_context={
                                "negative_role": "clean_negative_candidate_key",
                                "method_variant": "sstw_full_method",
                                "trial_index": public_trial_index,
                            },
                        )
                        public_scorer = _build_scorer(
                            pipelines[public_model_id],
                            calibrations[public_model_id],
                            replay_likelihood_configs[public_model_id],
                            prompt=public_prompt,
                            key_text=public_key,
                        )
                        public_video = _resolve_video(
                            root,
                            public_row.get("clean_negative_video_path"),
                            "videos",
                        )
                        public_result = optimize_bounded_parameter_attack_for_video(
                            public_video,
                            root / "adaptive_public_negative_probes" / str(source.get("statistical_cluster_id")),
                            attack_family=attack_family,
                            scorer=public_scorer,
                            objective="minimize_detector_score",
                            endpoint_reference=float(public_row["endpoint_score"]),
                            endpoint_tolerance=context["endpoint_tolerance"],
                            minimum_quality_psnr=context["minimum_quality_psnr"],
                            query_budget=context[
                                "public_negative_probe_query_budget"
                            ],
                            query_budget_checkpoints=(
                                1,
                                context["public_negative_probe_query_budget"],
                            ),
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
                        public_informed_strength = float(
                            public_result.selected.attack_parameters[
                                "attack_strength"
                            ]
                        )
                        public_probe_summary = {
                            "adaptive_attack_public_negative_probe_count": len(public_result.candidates),
                            "adaptive_attack_public_negative_cluster_id": public_row.get("statistical_cluster_id"),
                            "adaptive_attack_public_negative_candidate_records": [
                                row.as_dict() for row in public_result.candidates
                            ],
                            "adaptive_attack_public_negative_informed_strength": public_informed_strength,
                            "adaptive_attack_public_negative_query_budget_checkpoints": list(
                                (
                                    1,
                                    context[
                                        "public_negative_probe_query_budget"
                                    ],
                                )
                            ),
                            "adaptive_attack_public_negative_query_budget_checkpoint_records": list(
                                public_result.checkpoint_records()
                            ),
                        }
                    adaptive_output_dir = (
                        root
                        / "adaptive_attacked_videos"
                        / str(source.get("statistical_cluster_id"))
                        / protocol
                    )
                    if attack_family == "model_vae_regeneration":
                        result = optimize_model_vae_regeneration_attack_for_video(
                            pipeline,
                            source_video,
                            adaptive_output_dir,
                            scorer=scorer,
                            endpoint_reference=float(source["endpoint_score"]),
                            endpoint_tolerance=context["endpoint_tolerance"],
                            minimum_quality_psnr=context["minimum_quality_psnr"],
                            query_budget=context["query_budget"],
                            query_budget_checkpoints=context[
                                "query_budget_checkpoints"
                            ],
                        )
                        execution_backend = (
                            "model_vae_encode_perturb_decode_bounded_black_box_search"
                        )
                        optimizer_type = (
                            "sequential_detector_feedback_one_coordinate_model_vae_search"
                        )
                        search_policy = (
                            "boundary_seed_then_detector_feedback_interval_refinement"
                        )
                    else:
                        result = optimize_bounded_parameter_attack_for_video(
                            source_video,
                            adaptive_output_dir,
                            attack_family=attack_family,
                            scorer=scorer,
                            objective=objective,
                            endpoint_reference=float(source["endpoint_score"]),
                            endpoint_tolerance=context["endpoint_tolerance"],
                            minimum_quality_psnr=context["minimum_quality_psnr"],
                            query_budget=context["query_budget"],
                            query_budget_checkpoints=context[
                                "query_budget_checkpoints"
                            ],
                            initial_strength=public_informed_strength,
                        )
                        execution_backend = (
                            "detector_feedback_two_coordinate_continuous_pattern_search"
                        )
                        optimizer_type = (
                            "sequential_detector_feedback_two_coordinate_pattern_search"
                        )
                        search_policy = (
                            "base_and_axis_probes_then_detector_feedback_pattern_refinement"
                        )
                    selected = result.selected
                    payload = {
                        **base,
                        **result.as_dict(),
                        **public_probe_summary,
                        "adaptive_attack_execution_backend": execution_backend,
                        "adaptive_attack_optimizer_type": optimizer_type,
                        "adaptive_parameter_search_policy": search_policy,
                        "adaptive_attack_score": selected.detector_score,
                        "adaptive_attack_path_score": selected.path_score,
                        "adaptive_attack_endpoint_score": selected.endpoint_score,
                        "adaptive_attack_detected_by_sstw": selected.decision,
                        "adaptive_attack_score_semantics": "frozen_calibrated_flow_probability_posterior",
                        "adaptive_attack_total_detector_query_count": (
                            len(result.candidates)
                            + int(
                                public_probe_summary.get(
                                    "adaptive_attack_public_negative_probe_count"
                                )
                                or 0
                            )
                        ),
                        "adaptive_attack_query_accounting_protocol": (
                            "all_target_and_public_negative_frozen_detector_calls"
                        ),
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
                            len(retention_sources),
                        )
                        peer = retention_sources[peer_index]
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
                        "replay_likelihood_model_id": score_payload.get(
                            "replay_likelihood_model_id"
                        ),
                        "replay_likelihood_calibration_protocol": score_payload.get(
                            "replay_likelihood_calibration_protocol"
                        ),
                        "replay_likelihood_calibration_cluster_count": score_payload.get(
                            "replay_likelihood_calibration_cluster_count"
                        ),
                        "replay_relative_observation_noise_standard_deviation": score_payload.get(
                            "replay_relative_observation_noise_standard_deviation"
                        ),
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
    checkpoint_records = build_adaptive_query_budget_checkpoint_records(records)
    cluster_coverage = audit_adaptive_source_cluster_coverage(
        records,
        checkpoint_records,
        required_protocols=context["required_protocols"],
        minimum_source_cluster_count=(
            context["minimum_source_video_cluster_count_per_protocol"]
        ),
        minimum_spoof_source_cluster_count=(
            context["minimum_spoof_source_video_cluster_count"]
        ),
        query_budget_checkpoints=context["query_budget_checkpoints"],
        expected_retention_source_cluster_ids=retention_cluster_ids,
        expected_spoof_source_cluster_ids=spoof_cluster_ids,
    )
    checkpoint_statistics: list[dict[str, Any]] = []
    for protocol in sorted(ADAPTIVE_SEARCH_PROTOCOLS):
        for checkpoint in context["query_budget_checkpoints"]:
            scoped = [
                row for row in checkpoint_records
                if row.get("non_runtime_attack_protocol") == protocol
                and int(
                    row.get("adaptive_attack_query_budget_checkpoint") or 0
                )
                == int(checkpoint)
            ]
            if not scoped:
                continue
            estimate = clustered_binary_rate_interval(
                scoped,
                outcome_field="adaptive_attack_checkpoint_detected_by_sstw",
                purpose=f"adaptive_query_budget::{protocol}::q={checkpoint}",
            )
            checkpoint_statistics.append({
                "non_runtime_attack_protocol": protocol,
                "adaptive_attack_query_budget_checkpoint": int(checkpoint),
                "adaptive_attack_checkpoint_total_detector_query_count": int(
                    scoped[0][
                        "adaptive_attack_checkpoint_total_detector_query_count"
                    ]
                ),
                **estimate.as_dict(
                    "adaptive_attack_checkpoint_watermark_retention_rate"
                ),
            })
    for query in query_rows:
        query["adaptive_attack_query_statistical_role"] = (
            "dependent_repeated_query_within_source_video_cluster"
        )
    adaptive_query_provenance_ready = bool(query_rows) and all(
        query.get("video_sha256")
        and int(query.get("decoded_frame_count") or 0) > 0
        and query.get("detector_score_source") == FLOW_STATE_POSTERIOR_SCORE_SOURCE
        and query.get("frozen_final_score_threshold") is not None
        and query.get("threshold_source_split") == "calibration"
        and query.get("test_time_threshold_update_blocked") is True
        and query.get("replay_likelihood_model_id")
        == REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID
        and query.get("replay_likelihood_calibration_protocol")
        == "calibration_clean_video_null_residual_cluster_equal_mle"
        and int(query.get("replay_likelihood_calibration_cluster_count") or 0) >= 2
        and float(
            query.get("replay_relative_observation_noise_standard_deviation")
            or 0.0
        )
        > 0.0
        for query in query_rows
    )
    optimizer_evidence = audit_adaptive_optimizer_evidence(
        records,
        query_budget=context["query_budget"],
        query_budget_checkpoints=context["query_budget_checkpoints"],
        public_negative_probe_query_budget=context[
            "public_negative_probe_query_budget"
        ],
    )
    optimizer_evidence_ready = all(optimizer_evidence.values())
    expected_query_count = sum(
        int(record.get("adaptive_attack_query_count") or 0)
        for record in records
        if record.get("non_runtime_attack_protocol") not in CONTROL_FIELDS
    ) + len(public_probe_candidate_records)
    adaptive_execution_complete = (
        len(records) == expected_count
        and not failure_rows
        and adaptive_query_provenance_ready
        and optimizer_evidence_ready
        and len(query_rows) == expected_query_count
        and cluster_coverage[
            "adaptive_attack_source_cluster_coverage_ready"
        ]
        and cluster_coverage[
            "adaptive_attack_independent_unit_uniqueness_ready"
        ]
        and cluster_coverage[
            "adaptive_attack_query_budget_checkpoint_coverage_ready"
        ]
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
        "independent_video_count": len(retention_sources),
        "adaptive_spoof_independent_video_count": len(spoof_sources),
        "required_non_runtime_attack_protocols": list(context["required_protocols"]),
        "formal_adaptive_attack_execution_record_count": len(records),
        "formal_adaptive_attack_expected_record_count": expected_count,
        "formal_adaptive_attack_query_record_count": len(query_rows),
        "formal_adaptive_attack_expected_query_record_count": expected_query_count,
        "formal_adaptive_attack_failure_count": len(failure_rows),
        "adaptive_attack_candidate_query_count": len(adaptive_candidate_records),
        "adaptive_attack_public_negative_query_count": len(public_probe_candidate_records),
        "adaptive_attack_query_budget": context["query_budget"],
        "adaptive_attack_query_budget_checkpoints": list(
            context["query_budget_checkpoints"]
        ),
        "adaptive_attack_query_budget_checkpoint_protocol": (
            ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
        ),
        "adaptive_attack_query_budget_checkpoint_record_count": len(
            checkpoint_records
        ),
        "adaptive_attack_query_budget_statistics": checkpoint_statistics,
        "adaptive_attack_source_population_cluster_count": len(
            source_population
        ),
        "adaptive_attack_spoof_source_population_cluster_count": len(
            spoof_source_population
        ),
        **cluster_coverage,
        "adaptive_attack_query_provenance_decision": (
            "PASS" if adaptive_query_provenance_ready else "FAIL"
        ),
        "adaptive_detector_feedback_search_decision": (
            "PASS"
            if optimizer_evidence["adaptive_detector_feedback_search_ready"]
            else "FAIL"
        ),
        "adaptive_model_vae_regeneration_decision": (
            "PASS"
            if optimizer_evidence["adaptive_model_vae_regeneration_ready"]
            else "FAIL"
        ),
        "adaptive_public_negative_probe_decision": (
            "PASS"
            if optimizer_evidence["adaptive_public_negative_probe_ready"]
            else "FAIL"
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
    write_jsonl(
        root
        / "records"
        / "formal_adaptive_attack_query_budget_checkpoint_records.jsonl",
        checkpoint_records,
    )
    write_jsonl(root / "records" / "formal_adaptive_attack_failure_records.jsonl", failure_rows)
    write_csv(root / "tables" / "formal_adaptive_attack_execution_table.csv", records)
    write_csv(root / "tables" / "formal_adaptive_attack_query_table.csv", query_rows)
    write_csv(
        root
        / "tables"
        / "formal_adaptive_attack_query_budget_checkpoint_table.csv",
        checkpoint_records,
    )
    write_json(root / "artifacts" / "formal_adaptive_attack_execution_decision.json", audit)
    report_lines = [
        "# 正式逐视频 Adaptive Attack 执行报告",
        "",
        "该报告只汇总真实候选视频生成、冻结检测器查询和预注册查询预算前缀。",
        "较小预算 checkpoint 由同一次序贯搜索的真实前缀构建, 不使用外推或最终候选回填。",
        "",
        f"- 执行门禁: {audit['formal_adaptive_attack_execution_decision']}",
        f"- 独立 source-video cluster 数: {audit['independent_video_count']}",
        f"- 最大单视频查询预算: {audit['adaptive_attack_query_budget']}",
        "- 预注册查询预算 checkpoint: "
        + ", ".join(
            str(value) for value in audit["adaptive_attack_query_budget_checkpoints"]
        ),
        "- source-video cluster 覆盖: "
        + audit["adaptive_attack_source_cluster_coverage_decision"],
        "- 独立统计单位去重: "
        + audit["adaptive_attack_independent_unit_uniqueness_decision"],
        "- query-budget checkpoint 覆盖: "
        + audit["adaptive_attack_query_budget_checkpoint_coverage_decision"],
        "",
        "## Query-budget 曲线统计",
        "",
    ]
    for row in checkpoint_statistics:
        report_lines.append(
            "- "
            f"{row['non_runtime_attack_protocol']}, "
            f"target_q={row['adaptive_attack_query_budget_checkpoint']}, "
            "total_q="
            f"{row['adaptive_attack_checkpoint_total_detector_query_count']}: "
            "retention="
            f"{row['adaptive_attack_checkpoint_watermark_retention_rate_estimate']}, "
            "95% CI=["
            f"{row['adaptive_attack_checkpoint_watermark_retention_rate_ci_95_lower']}, "
            f"{row['adaptive_attack_checkpoint_watermark_retention_rate_ci_95_upper']}]"
        )
    report_path = root / "reports" / "formal_adaptive_attack_execution_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
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
