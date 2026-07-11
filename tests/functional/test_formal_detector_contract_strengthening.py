"""验证正式 Flow 后验拟合、完整 P6 门禁与冻结 artifact 契约。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import log

import numpy as np
import pytest

from main.methods.state_space_watermark.flow_state_posterior import (
    FLOW_STATE_ADMISSIBILITY_THRESHOLD_NAMES,
    FlowEvidenceObservation,
    POSTERIOR_FEATURE_NAMES,
    infer_flow_state_posterior,
)
from main.methods.state_space_watermark.formal_detector import (
    FlowDetectorMechanismConfig,
    _cross_fitted_conservative_score,
    _fit_admissibility_thresholds,
    _fit_linear_gaussian_state_model,
    fit_flow_detector_calibration,
    frozen_flow_detector_calibration_from_dict,
)


def _observation_sequence(
    base_score: float,
    *,
    replay_reliability: float = 0.9,
    coverage: float = 0.92,
) -> list[FlowEvidenceObservation]:
    """构造具有完整 phase 字段的轻量确定性观测序列。"""

    return [
        FlowEvidenceObservation(
            endpoint_score=base_score + 0.02 * index,
            velocity_score=base_score + 0.01 * index,
            path_score=base_score + 0.015 * index,
            path_endpoint_consistency=base_score,
            replay_log_likelihood_ratio=(base_score - 0.5) * 2.0,
            coverage_ratio=coverage,
            replay_reliability=replay_reliability,
            time_grid_reliability=0.95,
            flow_phase=index / 3.0,
        )
        for index in range(4)
    ]


def _heteroscedastic_matrix(offset: float, shock_scale: float) -> np.ndarray:
    """构造低可靠 phase 残差更大的状态序列。"""

    phases = np.linspace(0.0, 1.0, 6)
    reliabilities = np.asarray([1.0, 0.2, 1.0, 0.2, 1.0, 0.2])
    rows: list[np.ndarray] = []
    for index, (phase, reliability) in enumerate(zip(phases, reliabilities)):
        shock = (
            0.0
            if reliability == 1.0
            else shock_scale * (1.0 if index % 4 == 1 else -1.0)
        )
        values = np.asarray(
            [
                offset + (feature_index + 1) * 0.05 * index
                + shock * (1.0 + feature_index * 0.03)
                for feature_index in range(len(POSTERIOR_FEATURE_NAMES))
            ],
            dtype=np.float64,
        )
        values[POSTERIOR_FEATURE_NAMES.index("replay_reliability")] = reliability
        values[POSTERIOR_FEATURE_NAMES.index("flow_phase")] = phase
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.quick
def test_state_model_fits_phase_and_reliability_with_equal_video_group_weight() -> None:
    """phase 转移和异方差尺度必须由数据拟合且不受簇内复制放大。"""

    first = _heteroscedastic_matrix(0.1, 0.8)
    second = _heteroscedastic_matrix(-0.2, 1.0)
    means = np.zeros(len(POSTERIOR_FEATURE_NAMES), dtype=np.float64)
    scales = np.ones(len(POSTERIOR_FEATURE_NAMES), dtype=np.float64)
    base_model = _fit_linear_gaussian_state_model(
        [first, second],
        group_ids=["video-a", "video-b"],
        feature_means=means,
        feature_scales=scales,
    )
    duplicated_model = _fit_linear_gaussian_state_model(
        [first] * 8 + [second],
        group_ids=["video-a"] * 8 + ["video-b"],
        feature_means=means,
        feature_scales=scales,
    )
    weak_heteroscedastic_model = _fit_linear_gaussian_state_model(
        [
            _heteroscedastic_matrix(0.1, 0.1),
            _heteroscedastic_matrix(-0.2, 0.1),
        ],
        group_ids=["video-a", "video-b"],
        feature_means=means,
        feature_scales=scales,
    )

    assert len(base_model.phase_transition_matrix) == len(POSTERIOR_FEATURE_NAMES)
    assert any(
        abs(value) > 1e-6
        for row in base_model.phase_transition_matrix
        for value in row
    )
    assert base_model.phase_transition_reference == pytest.approx(0.5)
    assert base_model.reliability_observation_variance_scale > 1e-6
    assert base_model.reliability_observation_variance_scale > (
        weak_heteroscedastic_model.reliability_observation_variance_scale
    )
    assert np.asarray(duplicated_model.transition_matrix) == pytest.approx(
        np.asarray(base_model.transition_matrix)
    )
    assert np.asarray(duplicated_model.phase_transition_matrix) == pytest.approx(
        np.asarray(base_model.phase_transition_matrix)
    )
    assert duplicated_model.reliability_observation_variance_scale == pytest.approx(
        base_model.reliability_observation_variance_scale
    )


@pytest.mark.quick
def test_admissibility_thresholds_use_complete_positive_video_group_contract() -> None:
    """完整 P6 阈值必须只由正类独立视频簇的统计量决定。"""

    positives = [_observation_sequence(0.72), _observation_sequence(0.8)]
    negative_low = _observation_sequence(0.01, replay_reliability=0.01, coverage=0.01)
    negative_high = _observation_sequence(0.99, replay_reliability=1.0, coverage=1.0)
    labels = [1, 1, 0, 0]
    groups = ["positive-a", "positive-b", "negative-a", "negative-b"]
    probabilities = [0.9, 0.94, 0.5, 0.5]
    low_negative_thresholds = _fit_admissibility_thresholds(
        [*positives, negative_low, negative_low],
        probabilities,
        labels,
        groups,
    )
    high_negative_thresholds = _fit_admissibility_thresholds(
        [*positives, negative_high, negative_high],
        probabilities,
        labels,
        groups,
    )

    assert set(low_negative_thresholds) == set(
        FLOW_STATE_ADMISSIBILITY_THRESHOLD_NAMES
    )
    assert low_negative_thresholds == pytest.approx(high_negative_thresholds)
    assert low_negative_thresholds["endpoint_score"] > 0.7
    assert low_negative_thresholds["path_score"] > 0.7
    assert low_negative_thresholds["posterior_confidence"] >= 0.9


@pytest.mark.quick
def test_cross_fitted_score_enforces_every_p6_dimension_and_incomplete_context() -> None:
    """外层 OOF conservative score 必须与完整 P6 下界和熵上界同构。"""

    sequence = _observation_sequence(0.8)
    thresholds = _fit_admissibility_thresholds(
        [sequence, _observation_sequence(0.75)],
        [0.94, 0.9],
        [1, 1],
        ["positive-a", "positive-b"],
    )
    mechanism = FlowDetectorMechanismConfig(enforce_state_admissibility=True)
    assert _cross_fitted_conservative_score(
        sequence,
        0.94,
        thresholds,
        mechanism,
    ) == pytest.approx(0.94)

    low_field_values = {
        "endpoint_score": 0.0,
        "path_score": -1.0,
        "path_endpoint_consistency": 0.0,
        "coverage_ratio": 0.0,
        "replay_reliability": 0.0,
        "time_grid_reliability": 0.0,
    }
    for field_name, low_value in low_field_values.items():
        low_sequence = [
            replace(row, **{field_name: low_value})
            for row in sequence
        ]
        assert _cross_fitted_conservative_score(
            low_sequence,
            0.94,
            thresholds,
            mechanism,
        ) == 0.0
    confidence_only_thresholds = {
        **thresholds,
        "posterior_entropy_maximum": log(2.0),
    }
    assert _cross_fitted_conservative_score(
        sequence,
        0.5,
        confidence_only_thresholds,
        mechanism,
    ) == 0.0
    entropy_only_thresholds = {
        **thresholds,
        "posterior_confidence": 0.5,
    }
    assert _cross_fitted_conservative_score(
        sequence,
        0.5,
        entropy_only_thresholds,
        mechanism,
    ) == 0.0
    incomplete_thresholds = dict(thresholds)
    incomplete_thresholds.pop("path_endpoint_consistency")
    assert _cross_fitted_conservative_score(
        sequence,
        0.94,
        incomplete_thresholds,
        mechanism,
    ) == 0.0


def _complete_calibration_artifact() -> dict[str, object]:
    """拟合一个可由严格 loader 重建的最小完整 artifact。"""

    sequences = [
        _observation_sequence(0.2 + 0.01 * index)
        for index in range(4)
    ] + [
        _observation_sequence(0.7 + 0.01 * index)
        for index in range(4)
    ]
    calibration = fit_flow_detector_calibration(
        sequences,
        [0] * 4 + [1] * 4,
        [f"negative-{index}" for index in range(4)]
        + [f"positive-{index}" for index in range(4)],
        target_fpr=0.1,
    )
    return calibration.as_dict()


@pytest.mark.quick
def test_frozen_calibration_loader_fails_closed_on_incomplete_formal_contract() -> None:
    """旧协议、缺失 phase、零可靠性尺度或不完整 P6 均不得静默加载。"""

    artifact = _complete_calibration_artifact()
    restored = frozen_flow_detector_calibration_from_dict(artifact)
    assert restored.target_fpr == pytest.approx(0.1)

    missing_protocol = deepcopy(artifact)
    missing_protocol.pop("posterior_probability_calibration_protocol")
    with pytest.raises(ValueError, match="缺少正式字段"):
        frozen_flow_detector_calibration_from_dict(missing_protocol)

    missing_phase = deepcopy(artifact)
    del missing_phase["posterior_positive_state_space_model"][
        "phase_transition_matrix"
    ]
    with pytest.raises(ValueError, match="缺少正式字段"):
        frozen_flow_detector_calibration_from_dict(missing_phase)

    zero_reliability_scale = deepcopy(artifact)
    zero_reliability_scale["posterior_negative_state_space_model"][
        "reliability_observation_variance_scale"
    ] = 0.0
    with pytest.raises(ValueError, match="必须严格为正"):
        frozen_flow_detector_calibration_from_dict(zero_reliability_scale)

    incomplete_p6 = deepcopy(artifact)
    del incomplete_p6["posterior_admissibility_thresholds"]["endpoint_score"]
    with pytest.raises(ValueError, match="thresholds 不完整"):
        frozen_flow_detector_calibration_from_dict(incomplete_p6)

    legacy_protocol = deepcopy(artifact)
    legacy_protocol["posterior_probability_calibration_protocol"] = (
        "legacy_unspecified_probability_calibration"
    )
    with pytest.raises(ValueError, match="概率校准协议不兼容"):
        frozen_flow_detector_calibration_from_dict(legacy_protocol)


@pytest.mark.quick
def test_inference_p6_uses_same_raw_observation_summary_as_oof_threshold_fit() -> None:
    """推理准入不得把 OOF 原始观测阈值改用于不同口径的平滑隐状态。"""

    calibration = frozen_flow_detector_calibration_from_dict(
        _complete_calibration_artifact()
    )
    low_endpoint = [
        replace(row, endpoint_score=0.0)
        for row in _observation_sequence(0.8)
    ]

    posterior = infer_flow_state_posterior(
        low_endpoint,
        calibration.posterior_model,
    )

    assert posterior.endpoint_admissibility_measurement == 0.0
    assert "endpoint_score" in posterior.admissibility_failures
