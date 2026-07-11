"""验证 replay 高斯噪声只由 calibration clean-video residual 拟合。"""

from __future__ import annotations

import pytest

from main.methods.state_space_watermark.replay_inversion import (
    REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID,
    ReplayGaussianLikelihoodConfig,
    fit_replay_gaussian_likelihood_config,
    gaussian_replay_residual_likelihood,
)


@pytest.mark.quick
def test_replay_noise_fit_gives_each_video_cluster_equal_weight() -> None:
    """重复时间网格不得让某个视频簇获得更高的噪声拟合权重。"""

    compact = fit_replay_gaussian_likelihood_config(
        [0.01, 0.09],
        ["video-a", "video-b"],
    )
    repeated = fit_replay_gaussian_likelihood_config(
        [0.01, 0.01, 0.01, 0.09],
        ["video-a", "video-a", "video-a", "video-b"],
    )

    assert compact.relative_observation_noise_standard_deviation == pytest.approx(
        (0.05) ** 0.5
    )
    assert repeated.relative_observation_noise_standard_deviation == pytest.approx(
        compact.relative_observation_noise_standard_deviation
    )
    assert repeated.calibration_cluster_count == 2
    assert repeated.likelihood_model_id == REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID
    assert repeated.calibration_protocol == (
        "calibration_clean_video_null_residual_cluster_equal_mle"
    )


@pytest.mark.quick
def test_replay_noise_fit_rejects_pseudoreplicated_or_invalid_input() -> None:
    """噪声 calibration 必须包含至少2个独立视频簇和有限正残差。"""

    with pytest.raises(ValueError, match="至少需要2个独立"):
        fit_replay_gaussian_likelihood_config([0.01, 0.02], ["same", "same"])
    with pytest.raises(ValueError, match="有限正数"):
        fit_replay_gaussian_likelihood_config([0.01, float("nan")], ["a", "b"])
    with pytest.raises(ValueError, match="长度不一致"):
        fit_replay_gaussian_likelihood_config([0.01], ["a", "b"])


@pytest.mark.quick
def test_gaussian_replay_likelihood_uses_explicit_frozen_noise_config() -> None:
    """核心似然函数不得隐式恢复固定经验噪声比例。"""

    torch = pytest.importorskip("torch")
    observed = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    candidate = observed + 0.1
    null = observed + 0.2
    config = ReplayGaussianLikelihoodConfig(
        relative_observation_noise_standard_deviation=0.2,
        calibration_protocol=(
            "calibration_clean_video_null_residual_cluster_equal_mle"
        ),
        calibration_cluster_count=12,
    )

    likelihood = gaussian_replay_residual_likelihood(
        candidate,
        null,
        observed,
        config=config,
    )

    assert likelihood.observation_noise_variance == pytest.approx(0.04)
    assert likelihood.log_likelihood_ratio_per_dimension > 0.0
