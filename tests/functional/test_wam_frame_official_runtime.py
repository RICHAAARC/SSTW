"""验证 WAM-frame official runtime 的路径解析边界。"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from external_baseline.wam_frame_official_runtime import (
    WAMFrameOfficialRuntimeConfig,
    _detect_wam_score,
    _load_wam_model,
)


def _drop_fake_wam_modules() -> None:
    """清理测试中注入的官方源码模块缓存。"""

    for module_name in (
        "notebooks",
        "notebooks.inference_utils",
        "watermark_anything",
        "watermark_anything.data",
        "watermark_anything.data.metrics",
        "watermark_anything.data.transforms",
    ):
        sys.modules.pop(module_name, None)


@pytest.mark.quick
def test_wam_frame_loader_resolves_repo_relative_params_before_source_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bootstrap 写出的仓库相对 params 路径在切换 source cwd 后仍必须可用。"""

    pytest.importorskip("torch", reason="requires optional method-runtime dependency")

    repo_root = tmp_path / "repo"
    source_dir = repo_root / "external_baseline" / "primary" / "wam_frame" / "source"
    (source_dir / "notebooks").mkdir(parents=True)
    (source_dir / "watermark_anything" / "data").mkdir(parents=True)
    (source_dir / "checkpoints").mkdir(parents=True)
    (source_dir / "configs").mkdir(parents=True)
    (source_dir / "notebooks" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "watermark_anything" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "watermark_anything" / "data" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "watermark_anything" / "data" / "metrics.py").write_text(
        "def msg_predict_inference(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (source_dir / "watermark_anything" / "data" / "transforms.py").write_text(
        "default_transform = object()\nunnormalize_img = object()\n",
        encoding="utf-8",
    )
    (source_dir / "configs" / "embedder.yaml").write_text("model: fake\n", encoding="utf-8")
    (source_dir / "checkpoints" / "params.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "resources" / "external_baseline" / "wam_frame" / "wam_mit.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-wam-checkpoint")
    (source_dir / "notebooks" / "inference_utils.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "class _FakeModel:",
                "    def to(self, device):",
                "        self.device = device",
                "        return self",
                "    def eval(self):",
                "        return self",
                "def load_model_from_checkpoint(json_path, ckpt_path):",
                "    assert Path(json_path).is_file(), json_path",
                "    assert Path(ckpt_path).is_file(), ckpt_path",
                "    assert Path('configs/embedder.yaml').is_file()",
                "    return _FakeModel()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _drop_fake_wam_modules()
    monkeypatch.setenv(
        "SSTW_WAM_FRAME_PARAMS_PATH",
        "external_baseline/primary/wam_frame/source/checkpoints/params.json",
    )
    monkeypatch.setenv("SSTW_WAM_FRAME_CHECKPOINT_PATH", str(checkpoint))
    cwd_before = Path.cwd()

    model, transform, unnormalize_img, msg_predict_inference, device, checkpoint_path = _load_wam_model(
        WAMFrameOfficialRuntimeConfig(
            run_root=str(tmp_path / "run"),
            bundle_root=str(tmp_path / "bundle"),
            source_dir=str(source_dir),
            repo_root=str(repo_root),
            resource_root=str(tmp_path / "resources" / "external_baseline"),
            device="cpu",
        )
    )

    assert Path.cwd() == cwd_before
    assert device == "cpu"
    assert checkpoint_path == checkpoint.resolve()
    assert getattr(model, "device") == "cpu"
    assert transform is not None
    assert unnormalize_img is not None
    assert callable(msg_predict_inference)
    _drop_fake_wam_modules()


@pytest.mark.quick
def test_wam_frame_detect_passes_channel_mask_to_official_message_predictor() -> None:
    """WAM 官方 message predictor 需要 `[B, 1, H, W]` mask, 不能传 `[B, H, W]`。"""

    torch = pytest.importorskip("torch", reason="requires optional method-runtime dependency")

    class FakeWAMModel:
        """最小 fake model, 用于验证 WAM detect 输出的 mask 维度适配。"""

        def detect(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            frame_count, _channels, height, width = batch.shape
            preds = torch.zeros((frame_count, 33, height, width), dtype=torch.float32)
            preds[:, 0:1, :, :] = 3.0
            return {"preds": preds}

    def fake_transform(_image: object) -> torch.Tensor:
        """把 PIL 输入转成官方 transform 后的占位张量。"""

        return torch.zeros((3, 4, 4), dtype=torch.float32)

    def fake_msg_predict_inference(bit_preds: torch.Tensor, mask_preds: torch.Tensor) -> torch.Tensor:
        """断言 mask 保留 channel 维, 以覆盖 Colab 中出现的维度广播故障。"""

        assert bit_preds.shape == (2, 32, 4, 4)
        assert mask_preds.shape == (2, 1, 4, 4)
        return torch.ones((2, 32), dtype=torch.float32)

    video = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    message = torch.ones((1, 32), dtype=torch.float32)

    bit_accuracy, mask_confidence = _detect_wam_score(
        FakeWAMModel(),
        video,
        fake_transform,
        fake_msg_predict_inference,
        message,
        device="cpu",
        max_frames=8,
    )

    assert bit_accuracy == 1.0
    assert mask_confidence > 0.9
