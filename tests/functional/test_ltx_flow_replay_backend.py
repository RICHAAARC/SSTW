"""验证 LTX-Video 跨模型 Flow adapter 的轻量确定性语义。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from dataclasses import dataclass

import pytest
import torch

from main.methods.state_space_watermark.ltx_flow_replay_backend import (
    build_ltx_flow_schedule_points,
    encode_video_to_ltx_endpoint_latent,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _score_records_with_frozen_calibration,
)
from experiments.generative_video_model_probe.formal_adaptive_attack_executor import (
    _calibrations_by_model,
)
from experiments.generative_video_model_probe.formal_method_variants import (
    CLAIM2_PATH_NESTED_ABLATION_VARIANT,
    DETECTOR_ONLY_METHOD_VARIANTS,
    FORMAL_DETECTOR_VARIANTS,
    FORMAL_METHOD_VARIANTS,
    GENERATION_METHOD_VARIANTS,
)


class _LatentDistribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _FakeLTXVAE:
    dtype = torch.float32
    latents_mean = torch.zeros(4)
    latents_std = torch.ones(4)
    config = SimpleNamespace(scaling_factor=2.0)

    def encode(self, video: torch.Tensor) -> SimpleNamespace:
        del video
        canonical = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2, 1, 1)
        return SimpleNamespace(latent_dist=_LatentDistribution(canonical))


class _FakeLTXPipeline:
    _execution_device = torch.device("cpu")
    vae = _FakeLTXVAE()
    vae_temporal_compression_ratio = 8
    vae_spatial_compression_ratio = 32
    transformer_spatial_patch_size = 1
    transformer_temporal_patch_size = 1


class _FakeFlowMatchScheduler:
    config = {
        "base_image_seq_len": 1,
        "max_image_seq_len": 16,
        "base_shift": 0.5,
        "max_shift": 1.0,
    }

    def set_timesteps(self, num_inference_steps, device, sigmas, mu):
        del device
        self.received_step_count = num_inference_steps
        self.received_mu = mu
        self.timesteps = torch.arange(num_inference_steps, 0, -1, dtype=torch.float32)
        self.sigmas = torch.tensor([*sigmas, 0.0], dtype=torch.float32)


@pytest.mark.quick
def test_ltx_video_endpoint_is_normalized_and_packed_without_proxy(monkeypatch, tmp_path) -> None:
    """LTX endpoint 必须来自真实 VAE encode 结果并经过官方归一化与可逆 pack。"""

    video = torch.zeros((1, 3, 9, 32, 32), dtype=torch.float32)
    monkeypatch.setattr(
        "main.methods.state_space_watermark.ltx_flow_replay_backend.load_video_tensor_for_wan_vae",
        lambda _path, *, device, dtype: (video.to(device=device, dtype=dtype), 9),
    )

    packed, canonical, layout, metadata = encode_video_to_ltx_endpoint_latent(
        _FakeLTXPipeline(),
        tmp_path / "attacked.mp4",
    )

    expected = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2, 1, 1) * 2.0
    assert torch.equal(canonical, expected)
    assert torch.equal(layout.to_canonical(packed), expected)
    assert metadata["endpoint_evidence_source"] == "ltx_vae_reencoded_video_latent"
    assert metadata["flow_latent_layout_id"] == "ltx_packed_token_flow_latent"


@pytest.mark.quick
def test_ltx_replay_schedule_uses_sequence_length_shifted_sigma_grid() -> None:
    """LTX replay 必须复用官方 sequence-length shift, 不能套用 Wan 默认网格。"""

    scheduler = _FakeFlowMatchScheduler()
    layout = SimpleNamespace(num_frames=2, height=2, width=2)

    points = build_ltx_flow_schedule_points(
        scheduler,
        num_inference_steps=4,
        device=torch.device("cpu"),
        latent_layout=layout,
    )

    assert scheduler.received_step_count == 4
    assert scheduler.received_mu > scheduler.config["base_shift"]
    assert len(points) == 5
    assert points[0].sigma == pytest.approx(1.0)
    assert points[-1].sigma == pytest.approx(0.0)


@dataclass(frozen=True)
class _FakeCalibration:
    model_id: str
    method_variant: str

    def as_dict(self):
        return {
            "method_variant": self.method_variant,
            "posterior_calibration_brier_score": 0.1,
            "posterior_calibration_expected_calibration_error": 0.05,
            "posterior_calibration_group_count": 2,
        }


@pytest.mark.quick
def test_formal_detector_calibration_is_isolated_by_generation_model(monkeypatch) -> None:
    """Wan 与 LTX 的 calibration negative 不得混入同一个冻结后验或阈值。"""

    rows = [
        {
            "generation_model_id": model_id,
            "method_variant": method_variant,
            "split": "calibration",
            "sample_role": "clean_negative",
            "formal_flow_evidence_unit_id": (
                f"evidence::{model_id}::{method_variant}"
            ),
        }
        for model_id in ("wan", "ltx")
        for method_variant in GENERATION_METHOD_VARIANTS
    ]

    def fake_fit(records, *, method_variant, target_fpr):
        del target_fpr
        model_ids = {record["generation_model_id"] for record in records}
        assert len(model_ids) == 1
        return _FakeCalibration(model_ids.pop(), method_variant)

    def fake_apply(record, calibration):
        assert record["generation_model_id"] == calibration.model_id
        return {
            "S_final_conservative": 0.25,
            "decision": False,
            "target_fpr": 0.1,
        }

    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner.fit_flow_evidence_calibration",
        fake_fit,
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner.apply_frozen_flow_detector",
        fake_apply,
    )

    scored, threshold_records, calibrations = _score_records_with_frozen_calibration(
        rows,
        target_fpr=0.1,
    )

    assert len(scored) == 2 * len(FORMAL_METHOD_VARIANTS)
    assert len(threshold_records) == 2 * len(FORMAL_DETECTOR_VARIANTS)
    assert set(calibrations) == {
        (model_id, method_variant)
        for model_id in ("wan", "ltx")
        for method_variant in FORMAL_DETECTOR_VARIANTS
    }
    assert {record["generation_model_id"] for record in threshold_records} == {"wan", "ltx"}
    for model_id in ("wan", "ltx"):
        model_rows = [
            record
            for record in scored
            if record["generation_model_id"] == model_id
        ]
        assert {record["method_variant"] for record in model_rows} == set(
            FORMAL_METHOD_VARIANTS
        )
        detector_only_rows = [
            record
            for record in model_rows
            if record["method_variant"] in DETECTOR_ONLY_METHOD_VARIANTS
        ]
        assert len(detector_only_rows) == len(DETECTOR_ONLY_METHOD_VARIANTS)
        assert all(
            record["detector_only_ablation"] is True
            for record in detector_only_rows
        )
        assert all(
            record["detector_only_source_method_variant"] == "sstw_full_method"
            for record in detector_only_rows
        )
    nested_thresholds = [
        record
        for record in threshold_records
        if record["method_variant"]
        in {CLAIM2_PATH_NESTED_ABLATION_VARIANT, *DETECTOR_ONLY_METHOD_VARIANTS}
    ]
    assert nested_thresholds
    assert all(
        record["calibration_source_method_variant"] == "sstw_full_method"
        for record in nested_thresholds
    )


@pytest.mark.quick
def test_adaptive_attack_loads_model_specific_frozen_calibrations(
    tmp_path,
    monkeypatch,
) -> None:
    """Wan 与 LTX adaptive 查询不得误用同一份冻结后验或阈值。"""

    threshold_path = tmp_path / "thresholds" / "formal_flow_detector_thresholds.jsonl"
    threshold_path.parent.mkdir(parents=True)
    rows = [
        {
            "generation_model_id": model_id,
            "method_variant": "sstw_full_method",
            "threshold_source_split": "calibration",
            "test_time_threshold_update_blocked": True,
        }
        for model_id in ("wan", "ltx")
    ]
    threshold_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_adaptive_attack_executor."
        "frozen_flow_detector_calibration_from_governed_artifact",
        lambda row: f"calibration::{row['generation_model_id']}",
    )

    calibrations = _calibrations_by_model(tmp_path)

    assert calibrations == {
        "wan": "calibration::wan",
        "ltx": "calibration::ltx",
    }
