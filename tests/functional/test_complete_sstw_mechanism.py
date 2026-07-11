import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _base_record as _flow_evidence_base_record,
    _controlled_negative_records_from_positive,
    _generation_key,
    _paired_velocity_causal_records,
    _state_space_posterior_mechanism_failures,
)
from experiments.generative_video_model_probe.formal_adaptive_attack_executor import (
    _disjoint_collusion_peer_index,
)
from experiments.generative_video_model_probe.replay_and_sketch_gate import run_replay_and_sketch_gate
from evaluation.attacks.adaptive_video_optimizer import optimize_adaptive_attack_for_video
from evaluation.attacks.video_runtime_attack_protocol import FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
from main.methods.state_space_watermark.authenticated_trajectory_sketch import (
    build_authenticated_trajectory_sketch_payload,
    sign_authenticated_trajectory_sketch,
    verify_authenticated_trajectory_sketch,
)
from main.methods.state_space_watermark.endpoint_latent_detector import compute_endpoint_latent_evidence
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    build_flow_tubelet_key_direction_like,
)
from main.methods.state_space_watermark.flow_latent_layout import PackedTokenFlowLatentLayout
from main.methods.state_space_watermark.flow_velocity_runtime import FlowVelocityConstraintRuntime
from main.methods.state_space_watermark.formal_detector import (
    FLOW_STATE_POSTERIOR_SCORE_SOURCE,
    _fit_admissibility_thresholds,
    _fit_calibrated_posterior_model,
    observation_sequence_from_flow_evidence_record,
)
from experiments.generative_video_model_probe.formal_method_variants import (
    FORMAL_METHOD_VARIANTS,
    apply_frozen_flow_detector,
    fit_flow_evidence_calibration,
    observation_sequence_for_method_variant,
)
from main.methods.state_space_watermark.path_observation import compute_path_step_observation
from main.methods.state_space_watermark.replay_inversion import (
    FlowSchedulePoint,
    ReplayGaussianLikelihoodConfig,
    gaussian_replay_residual_likelihood,
    run_key_independent_inversion_hypothesis,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)
from main.methods.state_space_watermark.flow_state_posterior import (
    FLOW_STATE_POSTERIOR_MODEL_TYPE,
)
from main.methods.state_space_watermark.watermark_key_derivation import (
    WATERMARK_KEY_DERIVATION_ID,
    derive_watermark_key_text,
    derive_wrong_key_control_text,
)
from evaluation.protocol.paper_mechanism_contract import (
    audit_paper_profile_mechanism_contract,
    load_paper_mechanism_contract,
)
from evaluation.protocol.record_writer import write_json, write_jsonl


@pytest.mark.quick
def test_flow_evidence_identity_preserves_cross_model_role() -> None:
    """正式证据必须保留生成模型族与跨模型角色, 否则泛化审计会静默漏样本。"""

    record = _flow_evidence_base_record(
        {
            "generation_model_id": "cross-model",
            "generation_model_family": "rectified_flow_video",
            "cross_model_role": "cross_model_validation_model",
            "prompt_id": "prompt-a",
            "seed_id": "seed-a",
            "split": "test",
        },
        sample_role="attacked_positive",
        method_variant="sstw_full_method",
    )

    assert record["generation_model_family"] == "rectified_flow_video"
    assert record["cross_model_role"] == "cross_model_validation_model"


@pytest.mark.quick
def test_collusion_pairing_uses_disjoint_video_pairs() -> None:
    """collusion 的独立统计单元必须是互不重叠的视频对。"""

    assert [_disjoint_collusion_peer_index(index, 6) for index in range(6)] == [
        1,
        0,
        3,
        2,
        5,
        4,
    ]
    with pytest.raises(ValueError):
        _disjoint_collusion_peer_index(0, 5)


@pytest.mark.quick
def test_flow_tubelet_key_code_is_stable_and_key_conditioned() -> None:
    """生成、endpoint 与 replay 必须能复用同一确定性 tubelet key code。"""

    reference = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    first, first_metadata = build_flow_tubelet_key_direction_like(reference, key_text="owner-key-a")
    second, _ = build_flow_tubelet_key_direction_like(reference, key_text="owner-key-a")
    wrong, _ = build_flow_tubelet_key_direction_like(reference, key_text="owner-key-b")

    assert torch.allclose(first, second)
    assert not torch.allclose(first, wrong)
    assert torch.isclose(first.norm(), torch.tensor(1.0), atol=1e-6)
    assert first_metadata["flow_tubelet_count"] > 0


@pytest.mark.quick
def test_velocity_constraint_modifies_scheduler_model_output_with_budget() -> None:
    """弱约束必须修改 scheduler 消费的 model output, 且不突破范数预算。"""

    sample = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    model_output = torch.ones_like(sample)
    direction, _ = build_flow_tubelet_key_direction_like(sample, key_text="owner-key")

    constrained, record = apply_velocity_field_constraint(
        model_output,
        sample,
        direction,
        flow_phase=0.5,
    )

    assert not torch.allclose(constrained, model_output)
    assert record["velocity_field_source"] == "scheduler_model_output_before_flow_match_step"
    assert 0.0 < record["velocity_constraint_delta_ratio"] <= 0.02


@pytest.mark.quick
def test_path_observation_orients_velocity_to_descending_sigma_update() -> None:
    """Flow 递减 sigma 的 velocity 与实际位移必须映射到同一证据方向。"""

    sample = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    direction, _ = build_flow_tubelet_key_direction_like(sample, key_text="path-key")
    constrained_velocity = -direction
    sample_after = sample + 0.1 * direction

    observation = compute_path_step_observation(
        sample,
        sample_after,
        constrained_velocity,
        direction,
        flow_phase=0.5,
    )

    assert observation.path_projection_normalized > 0.99
    assert observation.velocity_projection_normalized > 0.99
    assert observation.path_velocity_consistency > 0.99


class FlowMatchFakeScheduler:
    """提供与 Diffusers Flow scheduler 相同的最小 step 接口。"""

    def step(self, model_output, timestep, sample, *args, **kwargs):
        del timestep, args, kwargs
        return (sample - 0.1 * model_output,)


@pytest.mark.quick
def test_flow_velocity_runtime_wraps_and_restores_scheduler_step() -> None:
    """运行时包装器必须记录真实更新并在退出 context 后恢复 scheduler。"""

    scheduler = FlowMatchFakeScheduler()
    original_step = scheduler.step
    sample = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    model_output = torch.ones_like(sample)
    with FlowVelocityConstraintRuntime(
        scheduler,
        key_text="owner-key",
        total_steps=3,
    ) as runtime:
        first = scheduler.step(model_output, torch.tensor(2.0), sample, return_dict=False)[0]
        scheduler.step(model_output, torch.tensor(1.0), first, return_dict=False)

    assert len(runtime.step_records) == 2
    assert runtime.step_records[1]["velocity_field_constraint_status"] == "applied"
    assert scheduler.step.__func__ is original_step.__func__


@pytest.mark.quick
def test_packed_token_flow_layout_roundtrip_is_exact() -> None:
    """LTX token latent 的 pack/unpack 必须保持每个元素和 tubelet 坐标不变。"""

    canonical = torch.arange(1 * 4 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 4, 3, 4, 6)
    layout = PackedTokenFlowLatentLayout(
        num_frames=3,
        height=4,
        width=6,
        spatial_patch_size=2,
        temporal_patch_size=1,
    )

    packed = layout.from_canonical(canonical)
    restored = layout.to_canonical(packed)

    assert packed.shape == (1, 18, 16)
    assert torch.equal(restored, canonical)
    assert layout.as_dict()["flow_latent_layout_roundtrip_exact"] is True


@pytest.mark.quick
def test_flow_velocity_runtime_applies_same_tubelet_primitive_to_packed_tokens() -> None:
    """三维 token 模型必须通过可逆布局使用同一个五维 SSTW tubelet 原语。"""

    scheduler = FlowMatchFakeScheduler()
    layout = PackedTokenFlowLatentLayout(
        num_frames=2,
        height=4,
        width=4,
        spatial_patch_size=1,
        temporal_patch_size=1,
    )
    canonical = torch.zeros((1, 3, 2, 4, 4), dtype=torch.float32)
    sample = layout.from_canonical(canonical)
    model_output = torch.ones_like(sample)

    with FlowVelocityConstraintRuntime(
        scheduler,
        key_text="ltx-owner-key",
        total_steps=3,
        latent_layout=layout,
    ) as runtime:
        first = scheduler.step(model_output, torch.tensor(2.0), sample, return_dict=False)[0]
        scheduler.step(model_output, torch.tensor(1.0), first, return_dict=False)

    assert runtime.endpoint_latent is not None
    assert runtime.canonical_endpoint_latent.shape == canonical.shape
    assert runtime.step_records[0]["flow_latent_layout_id"] == "packed_token_flow_latent"
    assert runtime.step_records[1]["velocity_field_constraint_status"] == "applied"


@pytest.mark.quick
def test_endpoint_latent_evidence_prefers_embedded_key() -> None:
    """沿正确 key direction 构造的 endpoint 必须优于错误 key。"""

    reference = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    direction, _ = build_flow_tubelet_key_direction_like(reference, key_text="embedded-key")
    endpoint = direction * 5.0 + torch.randn_like(direction) * 0.001

    matched = compute_endpoint_latent_evidence(endpoint, key_text="embedded-key")
    wrong = compute_endpoint_latent_evidence(endpoint, key_text="wrong-key")

    assert matched.score > wrong.score
    assert matched.projection > wrong.projection
    assert matched.coverage_ratio == 1.0


@pytest.mark.quick
def test_linear_flow_inversion_and_replay_closes_cycle() -> None:
    """常速度 Flow 的 reverse/forward 数值积分应恢复同一 endpoint。"""

    endpoint = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    schedule = [
        FlowSchedulePoint(timestep=2.0, sigma=1.0),
        FlowSchedulePoint(timestep=1.0, sigma=0.5),
        FlowSchedulePoint(timestep=0.0, sigma=0.0),
    ]

    def constant_velocity(latent, timestep, step_index):
        del timestep, step_index
        return torch.full_like(latent, 0.2)

    replay = run_key_independent_inversion_hypothesis(
        endpoint,
        schedule,
        constant_velocity,
        constant_velocity,
    )

    assert replay.reverse_step_count == 2
    assert replay.forward_step_count == 2
    assert replay.candidate_cycle_relative_error < 1e-6


@pytest.mark.quick
def test_replay_llr_is_gaussian_residual_likelihood_not_error_ratio() -> None:
    """候选 replay 的 LLR 必须来自同方差高斯残差概率模型。"""

    observed = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    candidate = observed + 0.01
    null = observed + 0.10

    likelihood = gaussian_replay_residual_likelihood(candidate, null, observed)

    expected = 0.5 * (
        likelihood.null_residual_mean_squared_error
        - likelihood.candidate_residual_mean_squared_error
    ) / likelihood.observation_noise_variance
    assert likelihood.log_likelihood_ratio_per_dimension == pytest.approx(expected)
    assert likelihood.log_likelihood_ratio_per_dimension > 0.0
    assert likelihood.candidate_log_likelihood_per_dimension > likelihood.null_log_likelihood_per_dimension
    assert likelihood.likelihood_model_id == (
        "endpoint_energy_scaled_isotropic_gaussian_per_latent_dimension"
    )


def _state_sequence(level: float, *, increasing: bool) -> list[dict[str, float]]:
    """构造轻量动态序列, 用于验证状态转移而不是 GPU 结果。"""

    rows = []
    for step_index in range(5):
        delta = 0.03 * step_index if increasing else -0.01 * step_index
        value = level + delta
        rows.append({
            "flow_phase": step_index / 4,
            "endpoint_score": value,
            "velocity_score": value - 0.05,
            "path_score": value - 0.03,
            "path_endpoint_consistency": value,
            "replay_log_likelihood_ratio": value - 0.4,
            "replay_reliability": 0.9,
            "time_grid_reliability": 0.9,
            "coverage_ratio": 1.0,
            "path_velocity_consistency": 0.9,
            "key_agnostic_endpoint_energy": 0.5,
            "key_agnostic_velocity_energy": 0.5,
            "key_agnostic_path_energy": 0.5,
        })
    return rows


@pytest.mark.quick
def test_flow_posterior_uses_fitted_state_transition_filter_and_smoother() -> None:
    """正式后验必须拟合 H0/H1 动力学并在完整序列上运行 filtering/smoothing。"""

    negatives = [
        {
            "sample_role": "clean_negative",
            "split": "calibration",
            "statistical_cluster_id": f"state-negative-{index}",
            "flow_state_observation_sequence": _state_sequence(0.2 + index * 0.001, increasing=False),
        }
        for index in range(20)
    ]
    positives = [
        {
            "sample_role": "attacked_positive",
            "split": "calibration",
            "statistical_cluster_id": f"state-positive-{index}",
            "flow_state_observation_sequence": _state_sequence(0.75 + index * 0.001, increasing=True),
        }
        for index in range(20)
    ]

    calibration = fit_flow_evidence_calibration(
        negatives + positives,
        method_variant="sstw_full_method",
        target_fpr=0.1,
    )
    positive_score = apply_frozen_flow_detector(positives[0], calibration)
    negative_score = apply_frozen_flow_detector(negatives[0], calibration)
    model_payload = calibration.posterior_model.as_dict()

    assert model_payload["posterior_model_type"].startswith(
        "dual_hypothesis_linear_gaussian_state_space_filter_rts_smoother"
    )
    assert model_payload["posterior_positive_state_space_model"]["training_transition_count"] > 0
    assert model_payload["posterior_negative_state_space_model"]["training_transition_count"] > 0
    assert calibration.posterior_probability_calibration_protocol == (
        "nested_source_video_group_cross_fitted_state_space_llr_and_platt"
    )
    assert calibration.posterior_probability_calibration_outer_fold_count >= 2
    assert calibration.posterior_probability_calibration_inner_fold_minimum >= 2
    assert calibration.fixed_fpr_threshold_score_source == (
        "outer_group_heldout_nested_cross_fitted_conservative_scores"
    )
    assert positive_score["flow_state_filter_step_count"] == 5
    assert positive_score["flow_state_filtering_status"] == "kalman_filter_ready"
    assert positive_score["flow_state_smoothing_status"] == "rauch_tung_striebel_smoother_ready"
    assert positive_score["flow_state_log_likelihood_ratio"] > negative_score["flow_state_log_likelihood_ratio"]
    assert positive_score["S_final_unconstrained"] > negative_score["S_final_unconstrained"]
    assert positive_score["flow_detector_score_source"] == FLOW_STATE_POSTERIOR_SCORE_SOURCE


@pytest.mark.quick
def test_formal_state_space_gate_rejects_single_step_or_unmeasured_posterior() -> None:
    """正式 Flow 门禁不得接受单步 fallback 或缺少动态转移的概率记录。"""

    negatives = [
        {
            "sample_role": "clean_negative",
            "split": "calibration",
            "statistical_cluster_id": f"gate-negative-{index}",
            "flow_state_observation_sequence": _state_sequence(0.2, increasing=False),
        }
        for index in range(12)
    ]
    positives = [
        {
            "sample_role": "attacked_positive",
            "split": "calibration",
            "statistical_cluster_id": f"gate-positive-{index}",
            "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
        }
        for index in range(12)
    ]
    calibration = fit_flow_evidence_calibration(
        negatives + positives,
        method_variant="sstw_full_method",
        target_fpr=0.1,
    )
    detection = apply_frozen_flow_detector(positives[0], calibration)
    valid_record = {
        **positives[0],
        **detection,
        "generation_model_id": "test-flow-model",
        "method_variant": "sstw_full_method",
        "formal_flow_evidence_unit_id": "formal-unit-a",
        "flow_state_observation_sequence_status": "measured_from_fixed_replay_path",
        "flow_state_observation_step_count": 5,
        "replay_likelihood_model_id": (
            "endpoint_energy_scaled_isotropic_gaussian_per_latent_dimension"
        ),
    }
    threshold_record = {
        "generation_model_id": "test-flow-model",
        "method_variant": "sstw_full_method",
        **calibration.as_dict(),
    }

    assert _state_space_posterior_mechanism_failures(
        [valid_record],
        [threshold_record],
    ) == []

    invalid_record = {
        **valid_record,
        "flow_state_observation_sequence": [valid_record["flow_state_observation_sequence"][0]],
        "flow_state_observation_sequence_status": "invalid_single_phase_input",
        "flow_state_observation_step_count": 1,
    }
    failures = _state_space_posterior_mechanism_failures(
        [invalid_record],
        [threshold_record],
    )

    assert failures
    assert "measured_state_observation_sequence" in failures[0]["failed_requirements"]
    assert "kalman_filter_consumed_complete_sequence" in failures[0]["failed_requirements"]


@pytest.mark.quick
def test_core_detector_rejects_missing_or_single_phase_observation() -> None:
    """核心检测器本身必须拒绝聚合替代值, 不能只依赖外层门禁补救。"""

    with pytest.raises(ValueError, match="至少2个真实 phase"):
        observation_sequence_from_flow_evidence_record({"endpoint_score": 0.8})
    with pytest.raises(ValueError, match="至少2个真实 phase"):
        observation_sequence_from_flow_evidence_record(
            {"flow_state_observation_sequence": _state_sequence(0.8, increasing=True)[:1]}
        )


@pytest.mark.quick
def test_flow_observation_preserves_explicit_zero_and_rejects_nonfinite_values() -> None:
    """规范字段中的0必须覆盖旧后备字段, NaN 不得进入概率模型。"""

    phase = {
        **_state_sequence(0.8, increasing=True)[0],
        "flow_phase": 0.0,
        "velocity_score": 0.0,
        "S_velocity": 0.9,
        "path_score": 0.0,
        "S_path_inv": 0.9,
        "coverage_ratio": 0.0,
        "endpoint_coverage_ratio": 1.0,
        "replay_reliability": 0.0,
        "replay_reliability_weight": 1.0,
        "replay_log_likelihood_ratio": 0.0,
        "replay_log_likelihood_ratio_mean": 2.0,
    }
    observation = observation_sequence_from_flow_evidence_record({
        "flow_state_observation_sequence": [phase, dict(phase)],
    })[0]

    assert observation.flow_phase == 0.0
    assert observation.velocity_score == 0.0
    assert observation.path_score == 0.0
    assert observation.coverage_ratio == 0.0
    assert observation.replay_reliability == 0.0
    assert observation.replay_log_likelihood_ratio == 0.0

    with pytest.raises(ValueError, match="有限数值"):
        observation_sequence_from_flow_evidence_record({
            "flow_state_observation_sequence": [
                {**phase, "endpoint_score": float("nan")},
                phase,
            ],
        })


@pytest.mark.quick
def test_replay_gaussian_likelihood_configuration_matches_method_contract() -> None:
    """核心 replay 概率模型默认值必须与受治理方法契约完全一致。"""

    core_method = json.loads(
        Path("configs/methods/sstw_core_method.json").read_text(
            encoding="utf-8"
        )
    )
    config = ReplayGaussianLikelihoodConfig()
    replay_config = core_method["replay_likelihood"]

    assert config.likelihood_model_id == replay_config["model_id"]
    assert config.minimum_observation_noise_variance == pytest.approx(
        replay_config["minimum_observation_noise_variance"]
    )
    assert config.relative_observation_noise_standard_deviation == pytest.approx(
        replay_config["relative_observation_noise_standard_deviation"]
    )


@pytest.mark.quick
def test_parameterized_core_defaults_match_minimal_method_config() -> None:
    """可抽离核心配置必须与代码默认参数一致, 防止实现与发布配置漂移。"""

    core_method = json.loads(
        Path("configs/methods/sstw_core_method.json").read_text(encoding="utf-8")
    )

    assert asdict(VelocityFieldConstraintConfig()) == core_method[
        "velocity_field_constraint"
    ]
    assert asdict(FlowTubeletKeyCodeConfig()) == core_method["flow_tubelet_key_code"]
    assert FLOW_STATE_POSTERIOR_MODEL_TYPE == core_method["flow_state_posterior"][
        "model_type"
    ]
    assert WATERMARK_KEY_DERIVATION_ID == core_method["watermark_key_derivation"][
        "algorithm_id"
    ]


@pytest.mark.quick
def test_watermark_direction_key_requires_owner_secret_and_record_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开的 model、prompt 和 seed 不得单独预测正式水印方向。"""

    secret = b"owner-secret-material-with-32-bytes-minimum"
    key_id = "owner-key-2026"
    monkeypatch.setenv("SSTW_TRAJECTORY_AUTHENTICATION_KEY", secret.decode("utf-8"))
    monkeypatch.setenv("SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID", key_id)
    record = {
        "generation_model_id": "flow-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "watermark_key_derivation_id": WATERMARK_KEY_DERIVATION_ID,
        "watermark_key_id": key_id,
    }

    derived = _generation_key(record)
    expected = derive_watermark_key_text(
        secret,
        key_id=key_id,
        generation_model_id="flow-model",
        prompt_id="prompt-a",
        seed_id="seed-a",
    )
    wrong_owner = derive_wrong_key_control_text(
        secret,
        key_id=key_id,
        generation_model_id="flow-model",
        prompt_id="prompt-a",
        seed_id="seed-a",
    )

    assert derived == expected
    assert wrong_owner != derived
    assert "flow-model::prompt-a::seed-a" not in derived
    with pytest.raises(RuntimeError, match="key ID"):
        _generation_key({**record, "watermark_key_id": "different-owner"})


@pytest.mark.quick
def test_state_space_calibration_is_source_video_group_equal_weighted() -> None:
    """重复同一视频的 key trial 不得改变状态模型或概率校准参数。"""

    negatives = [
        {
            "sample_role": "clean_negative",
            "split": "calibration",
            "statistical_cluster_id": f"equal-negative-{index}",
            "flow_state_observation_sequence": _state_sequence(
                0.2 + index * 0.01,
                increasing=False,
            ),
        }
        for index in range(8)
    ]
    positives = [
        {
            "sample_role": "attacked_positive",
            "split": "calibration",
            "statistical_cluster_id": f"equal-positive-{index}",
            "flow_state_observation_sequence": _state_sequence(
                0.7 + index * 0.01,
                increasing=True,
            ),
        }
        for index in range(8)
    ]
    base = fit_flow_evidence_calibration(
        negatives + positives,
        method_variant="sstw_full_method",
        target_fpr=0.1,
    )
    repeated = fit_flow_evidence_calibration(
        negatives + [dict(negatives[0]) for _ in range(9)] + positives,
        method_variant="sstw_full_method",
        target_fpr=0.1,
    )

    assert repeated.posterior_model.feature_means == pytest.approx(
        base.posterior_model.feature_means
    )
    assert torch.allclose(
        torch.tensor(
            repeated.posterior_model.negative_state_space_model.transition_matrix,
            dtype=torch.float64,
        ),
        torch.tensor(
            base.posterior_model.negative_state_space_model.transition_matrix,
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        torch.tensor(
            repeated.posterior_model.positive_state_space_model.transition_matrix,
            dtype=torch.float64,
        ),
        torch.tensor(
            base.posterior_model.positive_state_space_model.transition_matrix,
            dtype=torch.float64,
        ),
    )
    assert repeated.posterior_model.platt_slope == pytest.approx(
        base.posterior_model.platt_slope
    )
    assert repeated.posterior_model.platt_intercept == pytest.approx(
        base.posterior_model.platt_intercept
    )
    assert repeated.final_score_threshold == pytest.approx(base.final_score_threshold)
    assert repeated.calibration_negative_cluster_count == 8
    assert repeated.calibration_negative_count == 17


@pytest.mark.quick
def test_claim1_causal_pair_uses_same_detector_and_controlled_generation_state() -> None:
    """Claim-1 配对必须只改变速度约束, 并复用完整方法冻结检测器。"""

    calibration_rows = [
        {
            "sample_role": "clean_negative",
            "split": "calibration",
            "statistical_cluster_id": f"causal-negative-{index}",
            "flow_state_observation_sequence": _state_sequence(0.2, increasing=False),
        }
        for index in range(8)
    ] + [
        {
            "sample_role": "attacked_positive",
            "split": "calibration",
            "statistical_cluster_id": f"causal-positive-{index}",
            "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
        }
        for index in range(8)
    ]
    calibration = fit_flow_evidence_calibration(
        calibration_rows,
        method_variant="sstw_full_method",
        target_fpr=0.1,
    )
    common = {
        "generation_model_id": "test-flow-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "attack_name": "codec-a",
        "sample_role": "attacked_positive",
        "split": "test",
        "velocity_causal_pair_id": "causal-pair-a",
        "generation_seed_random": 17,
        "generation_generator_state_digest_random": "generator-state-a",
        "replay_sampler_signature": "FlowMatchEulerDiscreteScheduler:config-a",
        "authenticated_generation_time_grid_id": "grid-a",
        "statistical_cluster_id": "source-cluster-a",
    }
    full = {
        **common,
        "method_variant": "sstw_full_method",
        "velocity_causal_intervention_status": "velocity_constraint_enabled",
        "generation_source_video_sha256": "full-video-digest",
        "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
    }
    control = {
        **common,
        "method_variant": "without_velocity_constraint",
        "velocity_causal_intervention_status": "velocity_constraint_disabled",
        "generation_source_video_sha256": "control-video-digest",
        "flow_state_observation_sequence": _state_sequence(0.2, increasing=False),
    }

    paired = _paired_velocity_causal_records(
        [full, control],
        {("test-flow-model", "sstw_full_method"): calibration},
    )

    assert paired[0]["velocity_causal_pairing_status"] == "matched_single_intervention_design"
    assert paired[0]["paired_detector_method_variant"] == "sstw_full_method"
    assert paired[0]["paired_full_method_score"] >= 0.0
    assert paired[0]["paired_without_velocity_constraint_score"] >= 0.0

    mismatched = {**control, "generation_generator_state_digest_random": "different-state"}
    blocked = _paired_velocity_causal_records(
        [full, mismatched],
        {("test-flow-model", "sstw_full_method"): calibration},
    )
    assert blocked[0]["velocity_causal_pairing_status"] == (
        "blocked_by_unmatched_generation_design"
    )
    assert blocked[0]["metric_status"] == "missing"


@pytest.mark.quick
def test_wrong_condition_controls_become_real_negative_families() -> None:
    """negative family 必须来自真实错误假设, 不能按 trial 索引改名。"""

    positive = {
        "formal_flow_evidence_unit_id": "positive-unit-a",
        "generation_model_id": "test-flow-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "trajectory_trace_id": "trace-a",
        "split": "calibration",
        "method_variant": "sstw_full_method",
        "attack_name": "codec-a",
        "statistical_cluster_id": "source-cluster-a",
        "statistical_independent_unit": "source_video_prompt_seed",
        "replay_control_fixed_reverse_path_reused": True,
        "wrong_key_flow_state_observation_sequence": _state_sequence(0.2, increasing=False),
        "wrong_prompt_flow_state_observation_sequence": _state_sequence(0.25, increasing=False),
        "wrong_sampler_flow_state_observation_sequence": _state_sequence(0.3, increasing=False),
        "wrong_key_replay_log_likelihood_ratio": -0.3,
        "wrong_prompt_replay_log_likelihood_ratio": -0.2,
        "wrong_sampler_replay_log_likelihood_ratio": -0.1,
        "wrong_key_S_path_inv": 0.1,
        "wrong_prompt_S_path_inv": 0.1,
        "wrong_sampler_S_path_inv": 0.1,
    }

    records = _controlled_negative_records_from_positive(positive)

    assert {record["negative_family"] for record in records} == {
        "watermarked_video_wrong_key_hypothesis",
        "watermarked_video_wrong_prompt_hypothesis",
        "watermarked_video_wrong_sampler_time_grid_hypothesis",
    }
    assert all(record["sample_role"] == "controlled_negative" for record in records)
    assert all(
        record["replay_control_fixed_reverse_path_reused"] is True
        for record in records
    )
    assert all("clean_key_family" not in record["negative_family"] for record in records)


@pytest.mark.quick
def test_group_cross_fitting_keeps_paired_h0_h1_hypotheses_in_same_fold() -> None:
    """同一视频的正确与错误假设可有不同标签, 但不能跨 fold 泄漏。"""

    rows: list[dict[str, object]] = []
    for index in range(8):
        cluster_id = f"mixed-source-{index}"
        rows.extend([
            {
                "sample_role": "attacked_positive",
                "split": "calibration",
                "statistical_cluster_id": cluster_id,
                "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
            },
            {
                "sample_role": "controlled_negative",
                "split": "calibration",
                "statistical_cluster_id": cluster_id,
                "flow_state_observation_sequence": _state_sequence(0.2, increasing=False),
            },
        ])

    calibration = fit_flow_evidence_calibration(
        rows,
        method_variant="sstw_full_method",
        target_fpr=0.1,
    )

    assert calibration.calibration_positive_cluster_count == 8
    assert calibration.calibration_negative_cluster_count == 8
    assert calibration.posterior_model.calibration_group_count == 8


@pytest.mark.quick
def test_calibration_rejects_noncalibration_split_unknown_role_and_missing_cluster() -> None:
    """冻结后验和阈值不得静默吸收 held-out、未知标签或无簇记录。"""

    valid_rows = [
        {
            "sample_role": "clean_negative",
            "split": "calibration",
            "statistical_cluster_id": f"strict-negative-{index}",
            "flow_state_observation_sequence": _state_sequence(0.2, increasing=False),
        }
        for index in range(4)
    ] + [
        {
            "sample_role": "attacked_positive",
            "split": "calibration",
            "statistical_cluster_id": f"strict-positive-{index}",
            "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
        }
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="只能使用 calibration split"):
        fit_flow_evidence_calibration(
            [{**valid_rows[0], "split": "test"}, *valid_rows[1:]],
            method_variant="sstw_full_method",
            target_fpr=0.1,
        )
    with pytest.raises(ValueError, match="未知 sample_role"):
        fit_flow_evidence_calibration(
            [{**valid_rows[0], "sample_role": "unknown_negative"}, *valid_rows[1:]],
            method_variant="sstw_full_method",
            target_fpr=0.1,
        )
    missing_cluster = dict(valid_rows[0])
    missing_cluster.pop("statistical_cluster_id")
    with pytest.raises(KeyError, match="statistical_cluster_id"):
        fit_flow_evidence_calibration(
            [missing_cluster, *valid_rows[1:]],
            method_variant="sstw_full_method",
            target_fpr=0.1,
        )


@pytest.mark.quick
def test_outer_fold_admissibility_thresholds_exclude_validation_video_clusters() -> None:
    """阈值用的 conservative score 不得使用自身视频簇拟合准入阈值。"""

    records = [
        {
            "sample_role": "clean_negative",
            "statistical_cluster_id": f"admissibility-negative-{index}",
            "flow_state_observation_sequence": _state_sequence(0.2, increasing=False),
        }
        for index in range(8)
    ] + [
        {
            "sample_role": "attacked_positive",
            "statistical_cluster_id": f"admissibility-positive-{index}",
            "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
        }
        for index in range(8)
    ]
    labels = [0] * 8 + [1] * 8
    groups = [str(record["statistical_cluster_id"]) for record in records]
    base_sequences = [
        observation_sequence_from_flow_evidence_record(record)
        for record in records
    ]
    mutated_records = [dict(record) for record in records]
    for index in (0, 8):
        mutated_records[index] = {
            **mutated_records[index],
            "flow_state_observation_sequence": [
                {
                    **row,
                    "coverage_ratio": 0.01,
                    "replay_reliability": 0.01,
                    "time_grid_reliability": 0.01,
                }
                for row in mutated_records[index]["flow_state_observation_sequence"]
            ],
        }
    mutated_sequences = [
        observation_sequence_from_flow_evidence_record(record)
        for record in mutated_records
    ]

    base_fit = _fit_calibrated_posterior_model(base_sequences, labels, groups)
    mutated_fit = _fit_calibrated_posterior_model(mutated_sequences, labels, groups)

    for index in (0, 8):
        assert mutated_fit.nested_admissibility_thresholds[index] == pytest.approx(
            base_fit.nested_admissibility_thresholds[index]
        )
    assert (
        mutated_fit.model.admissibility_thresholds["coverage"]
        < base_fit.model.admissibility_thresholds["coverage"]
    )


@pytest.mark.quick
def test_positive_entropy_admissibility_ignores_negative_hypotheses_in_mixed_cluster() -> None:
    """正类熵上限只能汇总同簇正假设, 不能混入重复的负假设。"""

    sequence = observation_sequence_from_flow_evidence_record({
        "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
    })
    thresholds = _fit_admissibility_thresholds(
        [sequence, sequence, sequence, sequence, sequence],
        [0.99, 0.5, 0.5, 0.5, 0.99],
        [1, 0, 0, 0, 1],
        ["mixed", "mixed", "mixed", "mixed", "positive-only"],
    )

    assert thresholds["posterior_entropy_maximum"] < 0.1


@pytest.mark.quick
def test_generic_ssm_baseline_cannot_reuse_key_conditioned_sstw_observations() -> None:
    """generic SSM 必须只看无 key 轨迹能量, 不能复制完整 SSTW 特征。"""

    record = {
        "flow_state_observation_sequence": _state_sequence(0.8, increasing=True),
    }
    full = observation_sequence_for_method_variant(
        record,
        method_variant="sstw_full_method",
    )
    generic = observation_sequence_for_method_variant(
        record,
        method_variant="generic_ssm_baseline",
    )

    assert full[0].endpoint_score != generic[0].endpoint_score
    assert full[0].path_score != generic[0].path_score
    assert generic[0].endpoint_score == 0.5
    assert generic[0].velocity_score == 0.5
    assert generic[0].replay_log_likelihood_ratio == 0.0


@pytest.mark.quick
def test_candidate_key_cannot_change_fixed_inversion_observation() -> None:
    """不同候选 key 只能改变 forward hypothesis, 不能改变 reverse path。"""

    endpoint = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    schedule = [
        FlowSchedulePoint(timestep=2.0, sigma=1.0),
        FlowSchedulePoint(timestep=1.0, sigma=0.5),
        FlowSchedulePoint(timestep=0.0, sigma=0.0),
    ]

    def base_velocity(state, _timestep, _index):
        return torch.zeros_like(state)

    def candidate_a(state, _timestep, _index):
        return torch.full_like(state, 0.1)

    def candidate_b(state, _timestep, _index):
        return torch.full_like(state, -0.2)

    replay_a = run_key_independent_inversion_hypothesis(
        endpoint, schedule, base_velocity, candidate_a
    )
    replay_b = run_key_independent_inversion_hypothesis(
        endpoint, schedule, base_velocity, candidate_b
    )

    assert all(
        torch.equal(left, right)
        for left, right in zip(replay_a.reverse_states, replay_b.reverse_states)
    )
    assert not torch.equal(replay_a.forward_states[-1], replay_b.forward_states[-1])


@pytest.mark.quick
def test_authenticated_trajectory_sketch_rejects_tampering() -> None:
    """HMAC sketch 必须绑定 prompt、seed、模型、sampler、时间网格和路径观测。"""

    payload = build_authenticated_trajectory_sketch_payload(
        [{
            "trajectory_step_index": 0,
            "trajectory_timestep": 1.0,
            "flow_phase": 0.5,
            "path_projection_normalized": 0.1,
            "velocity_projection_normalized": 0.2,
            "path_velocity_consistency": 0.9,
        }],
        key_id="key-id",
        prompt_digest="prompt-digest",
        seed_id="seed-a",
        model_signature="wan-model",
        sampler_signature="FlowMatchEulerDiscreteScheduler",
        time_grid_id="grid-a",
        generation_nonce_random="nonce-a",
    )
    signed = sign_authenticated_trajectory_sketch(payload, authentication_key=b"secret")

    assert verify_authenticated_trajectory_sketch(signed, authentication_key=b"secret") is True
    tampered = json.loads(json.dumps(signed))
    tampered["trajectory_sketch_payload"]["seed_id"] = "seed-b"
    assert verify_authenticated_trajectory_sketch(tampered, authentication_key=b"secret") is False


@pytest.mark.quick
def test_fixed_fpr_calibration_never_exceeds_calibration_budget() -> None:
    """冻结阈值在 `>=` 判定下也不能超过 calibration 侧目标 FPR。"""

    negatives = [
        {
            "sample_role": "clean_negative",
            "split": "calibration",
            "statistical_cluster_id": f"negative-{index}",
            "flow_state_observation_sequence": _state_sequence(
                0.15 + index / 100.0,
                increasing=False,
            ),
        }
        for index in range(20)
    ]
    positives = [
        {
            **negatives[index],
            "sample_role": "attacked_positive",
            "statistical_cluster_id": f"positive-{index}",
            "flow_state_observation_sequence": _state_sequence(
                0.75 + index / 100.0,
                increasing=True,
            ),
        }
        for index in range(10)
    ]
    calibration = fit_flow_evidence_calibration(
        negatives + positives,
        method_variant="without_flow_state_admissibility",
        target_fpr=0.1,
    )
    decisions = [apply_frozen_flow_detector(record, calibration)["decision"] for record in negatives]

    assert sum(decisions) <= 2
    assert calibration.calibration_negative_count == 20


@pytest.mark.constraint
def test_all_paper_profiles_share_complete_mechanism_contract() -> None:
    """三个 paper profile 只允许统计强度与样本规模不同。"""

    contract = load_paper_mechanism_contract("configs/protocol/sstw_complete_paper_mechanism.json")
    configs = [
        json.loads(Path(f"configs/protocol/{profile}_generative_probe.json").read_text(encoding="utf-8"))
        for profile in ("probe_paper", "pilot_paper", "full_paper")
    ]

    audit = audit_paper_profile_mechanism_contract(configs, contract)

    assert audit.passed is True
    assert tuple(contract["formal_method_variants"]) == FORMAL_METHOD_VARIANTS
    assert audit.audited_profile_count == 3
    assert not audit.violations


@pytest.mark.quick
def test_full_claim3_gate_requires_real_replay_controls_and_hmac(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim-3 只有真实 replay、三类对照和 HMAC sketch 同时通过才能升级。"""

    run_root = tmp_path / "run"
    authentication_key = b"claim3-secret"
    monkeypatch.setenv("SSTW_TRAJECTORY_AUTHENTICATION_KEY", authentication_key.decode("utf-8"))
    payload = build_authenticated_trajectory_sketch_payload(
        [{
            "trajectory_step_index": 0,
            "trajectory_timestep": 1.0,
            "flow_phase": 0.5,
            "path_projection_normalized": 0.2,
            "velocity_projection_normalized": 0.3,
            "path_velocity_consistency": 0.9,
        }],
        key_id="owner-key-id",
        prompt_digest="prompt-digest",
        seed_id="seed-a",
        model_signature="wan-model",
        sampler_signature="FlowMatchEulerDiscreteScheduler",
        time_grid_id="grid-a",
        generation_nonce_random="nonce-a",
    )
    signed = sign_authenticated_trajectory_sketch(payload, authentication_key=authentication_key)
    write_jsonl(run_root / "records" / "trajectory_sketch_records.jsonl", [{
        "trajectory_trace_id": "trace-a",
        "generation_model_id": "wan-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "method_variant": "sstw_full_method",
        "authenticated_trajectory_sketch_status": "signed",
        **signed,
    }])
    write_jsonl(run_root / "records" / "formal_flow_evidence_records.jsonl", [{
        "sample_role": "attacked_positive",
        "method_variant": "sstw_full_method",
        "metric_status": "measured_formal",
        "trajectory_trace_id": "trace-a",
        "generation_model_id": "wan-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "attack_name": "codec-attack",
        "split": "test",
        "replay_inversion_status": "ready",
        "formal_flow_evidence_level": "attacked_video_key_independent_inversion_hypothesis_replay",
        "replay_trajectory_source": "attacked_video_endpoint_model_velocity_inversion",
        "replay_uncertainty_mean": 0.1,
        "replay_reliability_weight": 0.9,
        "replay_step_counts": [16, 20, 24],
        "replay_sampler_signature": "FlowMatchEulerDiscreteScheduler",
        "authenticated_generation_time_grid_id": "grid-a",
        "replay_prompt_digest": "prompt-digest",
        "wrong_key_control_margin": 0.2,
        "wrong_prompt_control_margin": 0.1,
        "wrong_sampler_control_margin": 0.1,
        "flow_state_admissibility_status": "pass",
        "flow_posterior_confidence": 0.9,
        "flow_watermark_posterior_probability": 0.9,
        "flow_watermark_posterior_log_odds": 2.197,
        "flow_state_posterior_entropy": 0.1,
        "flow_state_log_likelihood_ratio": 1.5,
        "flow_state_filter_step_count": 16,
        "flow_state_filtering_status": "kalman_filter_ready",
        "flow_state_smoothing_status": "rauch_tung_striebel_smoother_ready",
        "replay_log_likelihood_ratio_mean": 0.5,
        "replay_likelihood_model_id": "endpoint_energy_scaled_isotropic_gaussian_per_latent_dimension",
        "replay_control_fixed_reverse_path_reused": True,
        "flow_detector_score_source": FLOW_STATE_POSTERIOR_SCORE_SOURCE,
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
        "S_final_conservative": 0.8,
    }])
    write_json(run_root / "artifacts" / "three_layer_mechanism_evidence_decision.json", {
        "claim_1_velocity_constraint_detectable_watermark_decision": "PASS",
        "claim_2_path_evidence_independent_gain_decision": "PASS",
    })

    audit = run_replay_and_sketch_gate(run_root)

    assert audit["replay_and_sketch_gate_decision"] == "PASS"
    assert audit["claim3_full_support_allowed"] is True
    assert audit["replay_and_sketch_evidence_level"] == "attacked_video_key_independent_inversion_hypothesis_replay_with_hmac_sketch"
    complete = json.loads(
        (run_root / "artifacts" / "complete_paper_mechanism_claim_decision.json").read_text(encoding="utf-8")
    )
    assert complete["complete_paper_mechanism_claim_decision"] == "PASS"


@pytest.mark.quick
def test_adaptive_optimizer_generates_and_queries_candidates_per_video(tmp_path: Path) -> None:
    """adaptive optimizer 必须生成新视频并逐候选调用冻结 scorer。"""

    import imageio.v3 as iio
    import numpy as np

    source = tmp_path / "source.mp4"
    frames = [np.full((24, 32, 3), 80 + index, dtype=np.uint8) for index in range(6)]
    iio.imwrite(source, frames, fps=8)
    queried: list[Path] = []

    def scorer(path: Path) -> dict[str, object]:
        queried.append(path)
        score = 0.2 if "gaussian_blur" in path.name else 0.5
        return {
            "S_final_conservative": score,
            "endpoint_score": 0.8,
            "S_path_inv": score,
            "decision": score >= 0.4,
        }

    result = optimize_adaptive_attack_for_video(
        source,
        tmp_path / "candidates",
        candidate_attack_names=("gaussian_blur_runtime", "brightness_contrast_runtime"),
        scorer=scorer,
        objective="minimize_detector_score",
        endpoint_reference=0.8,
        minimum_quality_psnr=10.0,
    )

    assert len(queried) == 2
    assert result.selected.attack_name == "gaussian_blur_runtime"
    assert Path(result.selected.video_path).exists()
    assert result.selected.video_sha256
    assert result.selected.decoded_frame_count > 0
    assert result.query_budget == 2
