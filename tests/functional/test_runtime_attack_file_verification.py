"""验证正式 runtime attack 只能由真实文件级变换与效果验真产生。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.attacks.video_runtime_attack_protocol import (
    apply_runtime_attack_to_video_file,
    apply_runtime_attack_to_video_tensor,
)


pytestmark = pytest.mark.quick


def _write_random_video(path: Path, *, seed: int = 20260711) -> None:
    """写出含真实空间与时间变化的短视频, 避免 codec 测试退化为常量帧。"""

    import imageio.v3 as iio

    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=8.0, codec="libx264", output_params=["-crf", "18"])


def test_file_level_codec_attack_verifies_requested_codec_and_decoded_effect(
    tmp_path: Path,
) -> None:
    """codec 攻击必须保留 CRF 参数并证明重新解码后的像素序列已经变化。"""

    source = tmp_path / "source.mp4"
    output = tmp_path / "h264_crf33.mp4"
    _write_random_video(source)

    metadata = apply_runtime_attack_to_video_file(
        source,
        output,
        "h264_crf33_runtime",
        fps=8.0,
    )

    assert metadata["runtime_attack_formal_evidence_level"] == (
        "formal_runtime_video_transform_verified"
    )
    assert metadata["runtime_attack_proxy_free"] is True
    assert metadata["runtime_attack_effect_verified"] is True
    assert metadata["runtime_attack_decoded_effect_verified"] is True
    assert metadata["runtime_attack_requested_codec"] == "libx264"
    assert metadata["runtime_attack_observed_codec"] in {"h264", "avc", "avc1"}
    assert metadata["runtime_attack_requested_output_params"] == ["-crf", "33"]
    assert metadata["runtime_attack_writer_parameters_applied"] is True
    assert metadata["runtime_attack_output_file_changed"] is True
    assert float(metadata["runtime_attack_decoded_mean_absolute_error"]) > 0.0


def test_file_level_perspective_attack_is_not_roll_proxy(tmp_path: Path) -> None:
    """透视攻击必须改变帧像素摘要, 而不是用 tensor roll 冒充投影变换。"""

    source = tmp_path / "source.mp4"
    output = tmp_path / "perspective.mp4"
    _write_random_video(source, seed=7)

    metadata = apply_runtime_attack_to_video_file(
        source,
        output,
        "perspective_runtime",
        fps=8.0,
    )

    assert metadata["runtime_attack_preencode_effect_verified"] is True
    assert metadata["runtime_attack_decoded_effect_verified"] is True
    assert metadata["runtime_attack_source_frame_digest"] != metadata[
        "runtime_attack_transformed_frame_digest"
    ]
    assert metadata["runtime_attack_requested_codec"] is None


def test_tensor_codec_attack_fails_closed_instead_of_noop() -> None:
    """内存张量入口不得把 codec 或组合攻击写成已执行。"""

    torch = pytest.importorskip("torch")
    video = torch.zeros((4, 3, 16, 16), dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match="file_level_runtime_attack_required_for_codec_or_combined_attack",
    ):
        apply_runtime_attack_to_video_tensor(video, "h264_crf33_runtime")
    with pytest.raises(
        ValueError,
        match="file_level_runtime_attack_required_for_codec_or_combined_attack",
    ):
        apply_runtime_attack_to_video_tensor(
            video,
            "compression_crop_combined_runtime",
        )


def test_file_attack_rejects_output_aliasing_source(tmp_path: Path) -> None:
    """正式攻击不能覆盖输入文件后再把同一路径伪装成独立证据。"""

    source = tmp_path / "source.mp4"
    _write_random_video(source)

    with pytest.raises(ValueError, match="runtime_attack_output_must_not_overwrite_source"):
        apply_runtime_attack_to_video_file(
            source,
            source,
            "perspective_runtime",
            fps=8.0,
        )
