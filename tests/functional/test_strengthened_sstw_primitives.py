"""验证 P1、P4、P5、P6 与 P8 的正式核心语义。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from main.methods.state_space_watermark.authenticated_trajectory_sketch import (
    build_authenticated_trajectory_sketch_payload,
    sign_authenticated_trajectory_sketch,
    verify_authenticated_trajectory_sketch,
    verify_authenticated_trajectory_sketch_once,
)
from main.methods.state_space_watermark.flow_state_posterior import (
    POSTERIOR_FEATURE_NAMES,
    CalibratedFlowPosteriorModel,
    FlowEvidenceObservation,
    LinearGaussianFlowStateModel,
    infer_flow_state_posterior,
    posterior_feature_vector,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    FlowTubeletKeyContext,
    INDEPENDENT_BINARY_PAYLOAD_MODE,
    ZERO_BIT_KEY_PRESENCE_PAYLOAD_MODE,
    build_flow_tubelet_key_direction_like,
    build_integrated_flow_tubelet_key_direction_like,
)
from main.methods.state_space_watermark.path_observation import (
    aggregate_path_observations,
    compute_path_step_observation,
)


@pytest.mark.quick
def test_joint_tubelet_code_binds_prompt_sampler_phase_and_payload() -> None:
    """正式 P1 方向必须随生成上下文、phase 和独立 payload 改变。"""

    reference = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32)
    config = FlowTubeletKeyCodeConfig(
        temporal_size=2,
        spatial_height=2,
        spatial_width=2,
    )
    zero_bit = FlowTubeletKeyContext(
        prompt_digest="a" * 64,
        sampler_signature="flow-sampler-a",
    )
    phase_a, metadata_a = build_flow_tubelet_key_direction_like(
        reference,
        key_text="owner-key",
        config=config,
        flow_phase=0.25,
        key_context=zero_bit,
    )
    phase_b, metadata_b = build_flow_tubelet_key_direction_like(
        reference,
        key_text="owner-key",
        config=config,
        flow_phase=0.75,
        key_context=zero_bit,
    )
    other_sampler, _ = build_flow_tubelet_key_direction_like(
        reference,
        key_text="owner-key",
        config=config,
        flow_phase=0.25,
        key_context=replace(zero_bit, sampler_signature="flow-sampler-b"),
    )
    independent_payload = FlowTubeletKeyContext.independent_payload(
        prompt_digest="a" * 64,
        sampler_signature="flow-sampler-a",
        payload_bits=(1, 0, 1, 1),
    )
    payload_direction, payload_metadata = build_flow_tubelet_key_direction_like(
        reference,
        key_text="owner-key",
        config=config,
        flow_phase=0.25,
        key_context=independent_payload,
    )
    _compatibility, compatibility_metadata = build_flow_tubelet_key_direction_like(
        reference,
        key_text="owner-key",
        config=config,
    )

    assert not torch.allclose(phase_a, phase_b)
    assert not torch.allclose(phase_a, other_sampler)
    assert not torch.allclose(phase_a, payload_direction)
    assert metadata_a["flow_tubelet_formal_context_complete"] is True
    assert metadata_b["flow_tubelet_phase"] == pytest.approx(0.75)
    assert metadata_a["flow_payload_mode"] == ZERO_BIT_KEY_PRESENCE_PAYLOAD_MODE
    assert payload_metadata["flow_payload_mode"] == INDEPENDENT_BINARY_PAYLOAD_MODE
    assert payload_metadata["flow_payload_bit_count"] == 4
    assert compatibility_metadata["flow_tubelet_formal_context_complete"] is False


@pytest.mark.quick
def test_integrated_tubelet_code_is_deterministic_for_preregistered_grid() -> None:
    """同一预注册 phase 网格必须重建完全一致的 endpoint 参考方向。"""

    reference = torch.zeros((1, 1, 2, 4, 4), dtype=torch.float32)
    context = FlowTubeletKeyContext(
        prompt_digest="b" * 64,
        sampler_signature="flow-sampler",
    )
    first, first_metadata = build_integrated_flow_tubelet_key_direction_like(
        reference,
        key_text="owner-key",
        key_context=context,
        flow_phases=(0.25, 0.5, 0.75),
        integration_weights=(0.25, 0.5, 0.25),
    )
    second, second_metadata = build_integrated_flow_tubelet_key_direction_like(
        reference,
        key_text="owner-key",
        key_context=context,
        flow_phases=(0.25, 0.5, 0.75),
        integration_weights=(0.25, 0.5, 0.25),
    )

    assert torch.equal(first, second)
    assert first.norm().item() == pytest.approx(1.0)
    assert first_metadata["flow_tubelet_formal_context_complete"] is True
    assert first_metadata["flow_key_direction_digest"] == second_metadata[
        "flow_key_direction_digest"
    ]


@pytest.mark.quick
def test_delta_sigma_arc_length_path_quadrature_is_partition_stable() -> None:
    """同一直线路径的区间细分不得改变 P4 的归一化线积分。"""

    direction = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    direction = direction / direction.norm()
    start = torch.zeros_like(direction)
    end = direction.clone()
    whole = compute_path_step_observation(
        start,
        end,
        -direction,
        direction,
        flow_phase=0.5,
        delta_sigma=-1.0,
    ).as_dict()
    midpoint = direction * 0.4
    split = [
        compute_path_step_observation(
            start,
            midpoint,
            -direction,
            direction,
            flow_phase=0.25,
            delta_sigma=-0.4,
        ).as_dict(),
        compute_path_step_observation(
            midpoint,
            end,
            -direction,
            direction,
            flow_phase=0.75,
            delta_sigma=-0.6,
        ).as_dict(),
    ]
    whole_score = aggregate_path_observations([whole])
    split_score = aggregate_path_observations(split)
    weighted_score = aggregate_path_observations([
        {**row, "replay_reliability_weight": 0.5} for row in split
    ])

    assert whole_score["path_quadrature_context_complete"] is True
    assert split_score["path_quadrature_context_complete"] is True
    assert whole_score["S_path_inv"] == pytest.approx(1.0)
    assert split_score["S_path_inv"] == pytest.approx(1.0)
    assert split_score["path_total_sigma_measure"] == pytest.approx(1.0)
    assert split_score["path_total_arc_length"] == pytest.approx(1.0)
    assert weighted_score["S_path_inv"] == pytest.approx(0.5)


def _state_model(dimension: int) -> LinearGaussianFlowStateModel:
    """构造包含 phase 调制和 reliability 异方差的可重建状态模型。"""

    identity = tuple(
        tuple(0.5 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    phase_transition = tuple(
        tuple(0.1 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    covariance = tuple(
        tuple(0.1 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    return LinearGaussianFlowStateModel(
        transition_matrix=identity,
        transition_bias=tuple(0.0 for _ in range(dimension)),
        process_covariance=covariance,
        observation_covariance=covariance,
        initial_mean=tuple(0.0 for _ in range(dimension)),
        initial_covariance=covariance,
        training_sequence_count=4,
        training_group_count=4,
        training_transition_count=8,
        training_transition_group_count=4,
        phase_transition_matrix=phase_transition,
        phase_transition_reference=0.5,
        reliability_observation_variance_scale=2.0,
    )


def _observation(*, phase: float, path: float) -> FlowEvidenceObservation:
    """构造 P5/P6 所需的完整 phase 观测。"""

    return FlowEvidenceObservation(
        endpoint_score=0.8,
        velocity_score=0.6,
        path_score=path,
        path_endpoint_consistency=0.8,
        replay_log_likelihood_ratio=0.4,
        coverage_ratio=1.0,
        replay_reliability=0.8,
        time_grid_reliability=0.9,
        flow_phase=phase,
    )


@pytest.mark.quick
def test_phase_is_state_feature_and_state_model_roundtrip_preserves_dynamics() -> None:
    """P5 冻结 artifact 必须保留 phase transition 与 reliability 噪声参数。"""

    observation = _observation(phase=0.75, path=0.7)
    vector = posterior_feature_vector(observation)
    model = _state_model(len(POSTERIOR_FEATURE_NAMES))
    restored = LinearGaussianFlowStateModel.from_dict(model.as_dict())

    assert POSTERIOR_FEATURE_NAMES[-1] == "flow_phase"
    assert vector[-1] == pytest.approx(0.75)
    assert restored.phase_transition_matrix == model.phase_transition_matrix
    assert restored.phase_transition_reference == pytest.approx(0.5)
    assert restored.reliability_observation_variance_scale == pytest.approx(2.0)


@pytest.mark.quick
def test_admissibility_requires_endpoint_path_consistency_and_complete_contract() -> None:
    """P6 必须阻断路径证据不足或阈值合同不完整的候选。"""

    dimension = len(POSTERIOR_FEATURE_NAMES)
    state_model = _state_model(dimension)
    thresholds = {
        "endpoint_score": 0.2,
        "path_score": 0.2,
        "path_endpoint_consistency": 0.2,
        "posterior_confidence": 0.49,
        "coverage": 0.5,
        "replay_reliability": 0.5,
        "time_grid_reliability": 0.5,
        "posterior_entropy_maximum": 1.0,
    }
    calibrated = CalibratedFlowPosteriorModel(
        feature_names=POSTERIOR_FEATURE_NAMES,
        feature_means=tuple(0.0 for _ in range(dimension)),
        feature_scales=tuple(1.0 for _ in range(dimension)),
        negative_state_space_model=state_model,
        positive_state_space_model=state_model,
        platt_slope=1.0,
        platt_intercept=0.0,
        admissibility_thresholds=thresholds,
        calibration_brier_score=0.25,
        calibration_log_loss=0.69,
        calibration_expected_calibration_error=0.0,
        calibration_group_count=4,
        calibration_record_count=8,
    )
    phases = (0.25, 0.5, 0.75)
    accepted = infer_flow_state_posterior(
        [_observation(phase=phase, path=0.8) for phase in phases],
        calibrated,
    )
    rejected = infer_flow_state_posterior(
        [_observation(phase=phase, path=-0.8) for phase in phases],
        calibrated,
    )
    incomplete = infer_flow_state_posterior(
        [_observation(phase=phase, path=0.8) for phase in phases],
        replace(calibrated, admissibility_thresholds={}),
    )

    assert accepted.admissibility_context_complete is True
    assert accepted.admissible is True
    assert rejected.admissible is False
    assert "path_score" in rejected.admissibility_failures
    assert incomplete.admissibility_context_complete is False
    assert incomplete.admissible is False
    assert incomplete.conservative_score == 0.0


def _formal_sketch_payload() -> dict[str, object]:
    """构造绑定视频、记录和代码版本的最小正式 sketch。"""

    return build_authenticated_trajectory_sketch_payload(
        [{
            "trajectory_step_index": 0,
            "trajectory_timestep": 1.0,
            "flow_phase": 0.5,
            "path_projection_normalized": 0.2,
            "velocity_projection_normalized": 0.3,
            "path_velocity_consistency": 0.9,
        }],
        key_id="owner-a",
        prompt_digest="c" * 64,
        seed_id="seed-a",
        model_signature="model-a@commit",
        sampler_signature="sampler-a",
        time_grid_id="grid-a",
        generation_nonce_random="0123456789abcdef0123456789abcdef",
        trajectory_trace_id="trace-a",
        method_configuration_id="sstw_full_method",
        video_sha256="d" * 64,
        generation_record_digest="e" * 64,
        code_commit="f" * 40,
    )


@pytest.mark.quick
def test_authenticated_sketch_binds_lineage_and_rejects_nonce_replay() -> None:
    """P8 必须同时验证 HMAC、正式 lineage 绑定和 nonce 一次性消费。"""

    payload = _formal_sketch_payload()
    signed = sign_authenticated_trajectory_sketch(
        payload,
        authentication_key=b"owner-secret-material-with-32-bytes-minimum",
    )
    expected = {
        name: str(payload[name])
        for name in (
            "trajectory_trace_id",
            "method_configuration_id",
            "video_sha256",
            "generation_record_digest",
            "code_commit",
        )
    }
    consumed_nonces: set[str] = set()
    first = verify_authenticated_trajectory_sketch_once(
        signed,
        authentication_key=b"owner-secret-material-with-32-bytes-minimum",
        expected_binding=expected,
        consumed_nonces=consumed_nonces,
    )
    replayed = verify_authenticated_trajectory_sketch_once(
        signed,
        authentication_key=b"owner-secret-material-with-32-bytes-minimum",
        expected_binding=expected,
        consumed_nonces=consumed_nonces,
    )

    assert verify_authenticated_trajectory_sketch(
        signed,
        authentication_key=b"owner-secret-material-with-32-bytes-minimum",
    ) is True
    assert payload["trajectory_sketch_formal_binding_complete"] is True
    assert first.verified is True
    assert consumed_nonces == {"0123456789abcdef0123456789abcdef"}
    assert replayed.verified is False
    assert "nonce_replayed" in replayed.failure_reasons


@pytest.mark.quick
def test_authenticated_sketch_partial_binding_is_rejected() -> None:
    """只绑定 trace 而未绑定视频和代码版本不得生成看似正式的 sketch。"""

    with pytest.raises(ValueError, match="一次性完整提供"):
        build_authenticated_trajectory_sketch_payload(
            [{"trajectory_step_index": 0}],
            key_id="owner-a",
            prompt_digest="c" * 64,
            seed_id="seed-a",
            model_signature="model-a",
            sampler_signature="sampler-a",
            time_grid_id="grid-a",
            generation_nonce_random="nonce-a",
            trajectory_trace_id="trace-a",
        )
