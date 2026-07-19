"""验证 replay likelihood calibration 不会被第一条 source failure 截断。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    ReplayLikelihoodCalibrationError,
    _fit_model_specific_replay_likelihood_configs,
)


@pytest.mark.quick
def test_calibration_attempts_every_clean_source_before_failing(monkeypatch, tmp_path) -> None:
    attempted: list[str] = []

    def fake_replay(_pipeline, _video_path, **kwargs):
        prompt = str(kwargs["prompt"])
        attempted.append(prompt)
        if prompt == "prompt-0":
            raise RuntimeError("synthetic first-source OOM")
        return SimpleNamespace(
            endpoint_latent=torch.ones((1, 4, 1, 1, 1)),
            replay_trajectories=(
                SimpleNamespace(null_residual_mean_squared_error=0.25),
            ),
        )

    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner._run_attacked_video_replay_for_model",
        fake_replay,
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner._generation_key",
        lambda source, extra_context: f"key-{source['prompt_id']}-{extra_context['negative_role']}",
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner._validated_flow_key_context",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner._resolve_video_path",
        lambda *args, **kwargs: tmp_path / "clean.mp4",
    )
    clean = [
        {
            "split": "calibration",
            "generation_model_id": "Wan-AI/test",
            "prompt_id": f"p{index}",
            "seed_id": f"seed-{index}",
            "trajectory_trace_id": f"trace-{index}",
            "method_variant": "sstw_clean_unwatermarked_reference",
        }
        for index in range(3)
    ]
    prompt_map = {f"p{index}": f"prompt-{index}" for index in range(3)}
    pipeline = SimpleNamespace(scheduler=object())

    with pytest.raises(ReplayLikelihoodCalibrationError) as captured:
        _fit_model_specific_replay_likelihood_configs(
            tmp_path,
            clean,
            prompt_map,
            {"Wan-AI/test": pipeline},
            minimum_clean_video_cluster_count=2,
            calibration_replay_step_count=20,
        )

    assert attempted == ["prompt-0", "prompt-1", "prompt-2"]
    assert len(captured.value.failure_records) == 1
    assert captured.value.failure_records[0]["prompt_id"] == "p0"
