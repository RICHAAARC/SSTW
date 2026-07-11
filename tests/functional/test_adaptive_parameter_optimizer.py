"""验证逐视频 adaptive attack 的序贯连续参数搜索语义。"""

from __future__ import annotations

import pytest

from evaluation.attacks.adaptive_video_optimizer import (
    ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL,
    ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL,
    AdaptiveVideoCandidate,
    AdaptiveVideoOptimizationResult,
    _next_bounded_parameter,
    _next_two_coordinate_parameters,
    _parameterized_attack_frames,
)
from experiments.generative_video_model_probe.formal_adaptive_attack_executor import (
    audit_adaptive_optimizer_evidence,
)


def _candidate(parameter: float, score: float) -> AdaptiveVideoCandidate:
    """构造只用于优化决策测试的候选记录。"""

    return AdaptiveVideoCandidate(
        candidate_index=int(parameter * 10),
        attack_name="bounded_parameter_test",
        video_path="candidate.mp4",
        video_sha256="a" * 64,
        decoded_frame_count=4,
        quality_psnr=40.0,
        detector_score=score,
        detector_score_source=(
            "dual_hypothesis_state_space_marginal_likelihood_calibrated_probability_posterior"
        ),
        frozen_final_score_threshold=0.5,
        threshold_source_split="calibration",
        test_time_threshold_update_blocked=True,
        endpoint_score=0.7,
        path_score=score,
        decision=score >= 0.5,
        admissible=True,
        attack_parameters={"attack_strength": parameter},
    )


def _two_coordinate_candidate(
    candidate_index: int,
    coordinates: tuple[float, float],
    score: float,
    *,
    admissible: bool = True,
) -> AdaptiveVideoCandidate:
    """构造带完整二维搜索来源的反馈候选。"""

    return AdaptiveVideoCandidate(
        candidate_index=candidate_index,
        attack_name="two_coordinate_test",
        video_path=f"candidate_{candidate_index}.mp4",
        video_sha256="a" * 64,
        decoded_frame_count=4,
        quality_psnr=40.0 if admissible else 10.0,
        detector_score=score,
        detector_score_source="frozen_test_detector",
        frozen_final_score_threshold=0.5,
        threshold_source_split="calibration",
        test_time_threshold_update_blocked=True,
        endpoint_score=0.7,
        path_score=score,
        decision=score >= 0.5,
        admissible=admissible,
        attack_parameters={
            "attack_strength": float(sum(coordinates) / 2.0),
            "adaptive_search_coordinate_1_value": coordinates[0],
            "adaptive_search_coordinate_2_value": coordinates[1],
        },
        adaptive_search_protocol=ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL,
        adaptive_search_coordinate_1_name="coordinate_1",
        adaptive_search_coordinate_1_value=coordinates[0],
        adaptive_search_coordinate_2_name="coordinate_2",
        adaptive_search_coordinate_2_value=coordinates[1],
    )


@pytest.mark.quick
def test_bounded_search_uses_previous_detector_scores() -> None:
    """第三次查询必须围绕当前 detector 最优参数细化, 不能读取固定攻击列表。"""

    next_parameter = _next_bounded_parameter(
        [_candidate(0.0, 0.8), _candidate(1.0, 0.2)],
        [0.0, 1.0],
        objective="minimize_detector_score",
        lower_bound=0.0,
        upper_bound=1.0,
    )

    assert next_parameter == pytest.approx(0.5)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("attack_family", "first_parameter", "second_parameter"),
    [
        (
            "endpoint_path_perturbation",
            "temporal_blend_alpha",
            "temporal_offset_frames",
        ),
        ("public_detector_probe", "brightness_factor", "contrast_factor"),
        ("watermark_removal", "gaussian_blur_radius", "quantization_step"),
        ("detector_evasion", "crop_ratio", "noise_sigma"),
    ],
)
def test_each_attack_family_exposes_two_independent_native_parameters(
    attack_family: str,
    first_parameter: str,
    second_parameter: str,
) -> None:
    """固定任一坐标时, 另一个坐标只能改变其对应的原生攻击参数。"""

    import numpy as np

    frames = [
        np.full((16, 16, 3), 40 + 30 * index, dtype=np.uint8)
        for index in range(4)
    ]
    _, base_parameters = _parameterized_attack_frames(
        frames,
        attack_family=attack_family,
        strength=0.1,
        parameter_coordinates=(0.1, 0.1),
    )
    _, only_second_changed = _parameterized_attack_frames(
        frames,
        attack_family=attack_family,
        strength=0.5,
        parameter_coordinates=(0.1, 0.9),
    )
    _, only_first_changed = _parameterized_attack_frames(
        frames,
        attack_family=attack_family,
        strength=0.5,
        parameter_coordinates=(0.9, 0.1),
    )

    assert base_parameters[first_parameter] == pytest.approx(
        only_second_changed[first_parameter]
    )
    assert base_parameters[second_parameter] != only_second_changed[second_parameter]
    assert base_parameters[first_parameter] != only_first_changed[first_parameter]
    assert base_parameters[second_parameter] == pytest.approx(
        only_first_changed[second_parameter]
    )


@pytest.mark.quick
def test_two_coordinate_refinement_changes_with_detector_feedback() -> None:
    """第4次查询必须随前三次冻结检测器分数改变, 并记录反馈父候选。"""

    coordinates = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    first_coordinate_best = [
        _two_coordinate_candidate(0, coordinates[0], 0.8),
        _two_coordinate_candidate(1, coordinates[1], 0.1),
        _two_coordinate_candidate(2, coordinates[2], 0.5),
    ]
    second_coordinate_best = [
        _two_coordinate_candidate(0, coordinates[0], 0.8),
        _two_coordinate_candidate(1, coordinates[1], 0.5),
        _two_coordinate_candidate(2, coordinates[2], 0.1),
    ]

    first_proposal, first_phase, first_parent = _next_two_coordinate_parameters(
        first_coordinate_best,
        coordinates,
        objective="minimize_detector_score",
        initial_strength=None,
    )
    second_proposal, second_phase, second_parent = _next_two_coordinate_parameters(
        second_coordinate_best,
        coordinates,
        objective="minimize_detector_score",
        initial_strength=None,
    )

    assert first_proposal != second_proposal
    assert first_parent == 1
    assert second_parent == 2
    assert first_phase == second_phase == "detector_feedback_pattern_refinement"


@pytest.mark.quick
def test_two_coordinate_refinement_filters_inadmissible_low_score() -> None:
    """低分但不满足质量约束的候选不得成为后续搜索父候选。"""

    coordinates = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    candidates = [
        _two_coordinate_candidate(0, coordinates[0], 0.8),
        _two_coordinate_candidate(1, coordinates[1], 0.01, admissible=False),
        _two_coordinate_candidate(2, coordinates[2], 0.3),
    ]

    _, _, parent = _next_two_coordinate_parameters(
        candidates,
        coordinates,
        objective="minimize_detector_score",
        initial_strength=None,
    )

    assert parent == 2


@pytest.mark.quick
def test_adaptive_evidence_requires_true_vae_and_public_feedback_queries() -> None:
    """正式 adaptive 门禁不得把固定变换列表或普通转码记作模型重生成。"""

    def candidates(*, vae: bool) -> list[dict]:
        rows = []
        coordinate_pairs = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        query_phases = ("base_point", "coordinate_1_probe", "coordinate_2_probe")
        for index, strength in enumerate((0.0, 1.0, 0.5)):
            parameters = {"attack_strength": strength}
            if vae:
                parameters.update({
                    "model_vae_regeneration_status": (
                        "measured_model_vae_encode_perturb_decode"
                    ),
                    "model_vae_class": "RealVideoVAE",
                    "model_vae_noise_direction_policy": (
                        "fixed_per_source_video_across_strength_queries"
                    ),
                    "model_vae_source_frame_count": 8,
                    "model_vae_output_frame_count": 8,
                })
            row = {
                "candidate_index": index,
                "video_sha256": f"digest-{index}",
                "detector_score": 0.8 - 0.2 * index,
                "path_score": 0.7 - 0.1 * index,
                "decision": index < 2,
                "admissible": True,
                "attack_parameters": parameters,
            }
            if not vae:
                row.update({
                    "adaptive_search_protocol": (
                        ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL
                    ),
                    "adaptive_search_query_phase": query_phases[index],
                    "adaptive_search_coordinate_1_value": coordinate_pairs[index][0],
                    "adaptive_search_coordinate_2_value": coordinate_pairs[index][1],
                })
            rows.append(row)
        return rows

    def checkpoints(rows: list[dict]) -> list[dict]:
        selected = min(
            rows,
            key=lambda row: (
                row["detector_score"],
                row["path_score"],
                row["candidate_index"],
            ),
        )
        return [{
            "adaptive_attack_query_budget_checkpoint": 3,
            "adaptive_attack_checkpoint_observed_query_count": 3,
            "adaptive_attack_checkpoint_candidate_count": 3,
            "adaptive_attack_checkpoint_admissible_candidate_count": 3,
            "adaptive_attack_checkpoint_has_admissible_candidate": True,
            "adaptive_attack_checkpoint_selection_protocol": (
                ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
            ),
            "adaptive_attack_checkpoint_selected_candidate_index": selected[
                "candidate_index"
            ],
            "adaptive_attack_checkpoint_output_video_sha256": selected[
                "video_sha256"
            ],
            "adaptive_attack_checkpoint_detector_score": selected[
                "detector_score"
            ],
            "adaptive_attack_checkpoint_detected_by_sstw": selected["decision"],
        }]

    regeneration_candidates = candidates(vae=True)
    public_candidates = candidates(vae=False)

    regeneration = {
        "non_runtime_attack_protocol": (
            "generative_recompression_or_regeneration_attack"
        ),
        "adaptive_attack_optimizer_type": (
            "sequential_detector_feedback_one_coordinate_model_vae_search"
        ),
        "adaptive_attack_query_count": 3,
        "adaptive_attack_total_detector_query_count": 3,
        "adaptive_attack_query_accounting_protocol": (
            "all_target_and_public_negative_frozen_detector_calls"
        ),
        "adaptive_attack_objective": "minimize_detector_score",
        "adaptive_attack_selected_parameters": {"attack_strength": 0.5},
        "adaptive_attack_candidate_records": regeneration_candidates,
        "adaptive_attack_query_budget_checkpoint_protocol": (
            ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
        ),
        "adaptive_attack_query_budget_checkpoints": [3],
        "adaptive_attack_query_budget_checkpoint_records": checkpoints(
            regeneration_candidates
        ),
    }
    public_probe = {
        "non_runtime_attack_protocol": "detector_probing_with_public_negatives",
        "adaptive_attack_optimizer_type": (
            "sequential_detector_feedback_two_coordinate_pattern_search"
        ),
        "adaptive_search_protocol": ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL,
        "adaptive_attack_query_count": 3,
        "adaptive_attack_total_detector_query_count": 6,
        "adaptive_attack_query_accounting_protocol": (
            "all_target_and_public_negative_frozen_detector_calls"
        ),
        "adaptive_attack_objective": "minimize_detector_score",
        "adaptive_attack_selected_parameters": {"attack_strength": 0.5},
        "adaptive_attack_candidate_records": public_candidates,
        "adaptive_attack_query_budget_checkpoint_protocol": (
            ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
        ),
        "adaptive_attack_query_budget_checkpoints": [3],
        "adaptive_attack_query_budget_checkpoint_records": checkpoints(
            public_candidates
        ),
        "adaptive_attack_public_negative_probe_count": 3,
        "adaptive_attack_public_negative_candidate_records": public_candidates,
        "adaptive_attack_public_negative_query_budget_checkpoints": [3],
        "adaptive_attack_public_negative_query_budget_checkpoint_records": (
            checkpoints(public_candidates)
        ),
        "adaptive_attack_public_negative_informed_strength": 0.5,
    }

    ready = audit_adaptive_optimizer_evidence(
        [regeneration, public_probe],
        query_budget=3,
    )
    blocked = audit_adaptive_optimizer_evidence(
        [
            {
                **regeneration,
                "adaptive_attack_candidate_records": candidates(vae=False),
            },
            public_probe,
        ],
        query_budget=3,
    )

    assert all(ready.values())
    assert blocked["adaptive_model_vae_regeneration_ready"] is False


@pytest.mark.quick
def test_query_budget_checkpoints_use_only_actual_nested_query_prefixes() -> None:
    """较小 budget 不得复用尚未发生的最终查询候选。"""

    candidates = (
        _candidate(0.0, 0.8),
        _candidate(0.1, 0.3),
        _candidate(0.2, 0.5),
        _candidate(0.3, 0.1),
    )
    result = AdaptiveVideoOptimizationResult(
        objective="minimize_detector_score",
        selected=candidates[-1],
        candidates=candidates,
        endpoint_reference=0.7,
        endpoint_tolerance=0.08,
        minimum_quality_psnr=24.0,
        query_budget=4,
        query_budget_checkpoints=(1, 3, 4),
    )

    checkpoint_records = result.checkpoint_records()

    assert [
        row["adaptive_attack_checkpoint_selected_candidate_index"]
        for row in checkpoint_records
    ] == [0, 1, 3]
    assert [
        row["adaptive_attack_checkpoint_observed_query_count"]
        for row in checkpoint_records
    ] == [1, 3, 4]
    assert all(
        row["adaptive_attack_checkpoint_selection_protocol"]
        == ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
        for row in checkpoint_records
    )
