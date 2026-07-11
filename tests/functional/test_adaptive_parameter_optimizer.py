"""验证逐视频 adaptive attack 的序贯连续参数搜索语义。"""

from __future__ import annotations

import pytest

from evaluation.attacks.adaptive_video_optimizer import (
    AdaptiveVideoCandidate,
    _next_bounded_parameter,
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
def test_parameterized_evasion_changes_real_frames_continuously() -> None:
    """不同连续强度必须产生不同真实视频帧和显式参数。"""

    import numpy as np

    frames = [np.full((16, 16, 3), 128, dtype=np.uint8) for _ in range(4)]
    weak, weak_parameters = _parameterized_attack_frames(
        frames,
        attack_family="detector_evasion",
        strength=0.2,
    )
    strong, strong_parameters = _parameterized_attack_frames(
        frames,
        attack_family="detector_evasion",
        strength=0.8,
    )

    assert weak_parameters["crop_ratio"] > strong_parameters["crop_ratio"]
    assert weak_parameters["noise_sigma"] < strong_parameters["noise_sigma"]
    assert not np.array_equal(weak[0], strong[0])


@pytest.mark.quick
def test_adaptive_evidence_requires_true_vae_and_public_feedback_queries() -> None:
    """正式 adaptive 门禁不得把固定变换列表或普通转码记作模型重生成。"""

    def candidates(*, vae: bool) -> list[dict]:
        rows = []
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
            rows.append({
                "candidate_index": index,
                "attack_parameters": parameters,
            })
        return rows

    regeneration = {
        "non_runtime_attack_protocol": (
            "generative_recompression_or_regeneration_attack"
        ),
        "adaptive_attack_optimizer_type": (
            "sequential_detector_feedback_bounded_parameter_search"
        ),
        "adaptive_attack_query_count": 3,
        "adaptive_attack_selected_parameters": {"attack_strength": 0.5},
        "adaptive_attack_candidate_records": candidates(vae=True),
    }
    public_probe = {
        "non_runtime_attack_protocol": "detector_probing_with_public_negatives",
        "adaptive_attack_optimizer_type": (
            "sequential_detector_feedback_bounded_parameter_search"
        ),
        "adaptive_attack_query_count": 3,
        "adaptive_attack_selected_parameters": {"attack_strength": 0.5},
        "adaptive_attack_candidate_records": candidates(vae=False),
        "adaptive_attack_public_negative_probe_count": 3,
        "adaptive_attack_public_negative_candidate_records": candidates(vae=False),
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
