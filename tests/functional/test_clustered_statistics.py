"""验证论文统计单位固定为独立 source video。"""

from __future__ import annotations

import pytest

from evaluation.statistics.clustered_inference import (
    clustered_binary_any_rate_interval,
    clustered_mean_interval,
    one_sided_binomial_upper_bound,
)


@pytest.mark.quick
def test_repeating_key_trials_does_not_change_cluster_equal_weight_estimate() -> None:
    """复制同一视频的 key trial 不得改变总体点估计或独立样本数。"""

    original = clustered_mean_interval(
        {"video-a": [1.0], "video-b": [0.0]},
        bootstrap_resamples=500,
        purpose="pseudoreplication-original",
    )
    repeated = clustered_mean_interval(
        {"video-a": [1.0] * 100, "video-b": [0.0] * 100},
        bootstrap_resamples=500,
        purpose="pseudoreplication-repeated",
    )

    assert original.estimate == repeated.estimate == 0.5
    assert original.cluster_count == repeated.cluster_count == 2
    assert repeated.observation_count == 200


@pytest.mark.quick
def test_exact_fpr_bound_uses_independent_video_count() -> None:
    """30个独立零误报视频应形成约0.1的单侧 95% 上界。"""

    upper = one_sided_binomial_upper_bound(0, 30)
    assert 0.09 < upper < 0.10


@pytest.mark.quick
def test_multiple_negative_hypotheses_cannot_dilute_video_level_fpr() -> None:
    """同一视频增加正确拒绝的负假设, 不得稀释已发生的误接受。"""

    records = [
        {"statistical_cluster_id": "video-a", "decision": True},
        *[
            {"statistical_cluster_id": "video-a", "decision": False}
            for _ in range(99)
        ],
        {"statistical_cluster_id": "video-b", "decision": False},
    ]

    estimate = clustered_binary_any_rate_interval(
        records,
        outcome_field="decision",
        bootstrap_resamples=500,
        purpose="negative-hypothesis-fpr",
    )

    assert estimate.estimate == 0.5
    assert estimate.cluster_count == 2
