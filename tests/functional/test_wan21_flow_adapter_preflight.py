"""验证 Wan2.1 Flow adapter preflight 的轻量接口结构。"""

from __future__ import annotations

import pytest

from experiments.flow_model_adapter_preflight.wan21_preflight import WAN21_PRIMARY_MODEL_ID, _sampler_signature


class _DummyScheduler:
    """用于测试 sampler signature 的最小 scheduler 对象。"""

    config = {"num_train_timesteps": 1000, "prediction_type": "flow_prediction"}


class _DummyPipe:
    """用于测试 sampler signature 的最小 pipeline 对象。"""

    scheduler = _DummyScheduler()


@pytest.mark.quick
def test_sampler_signature_is_stable_for_same_scheduler() -> None:
    """相同模型 ID 和 scheduler 配置必须得到稳定 sampler signature。"""
    first = _sampler_signature(_DummyPipe(), WAN21_PRIMARY_MODEL_ID)
    second = _sampler_signature(_DummyPipe(), WAN21_PRIMARY_MODEL_ID)
    assert first == second
    assert first["sampler_signature_id"].startswith("sampler_signature_")
    assert len(first["sampler_signature_sha256"]) == 64
