"""验证 REVMark official runtime 的轻量路径适配。"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from external_baseline.revmark_official_runtime import (
    REVMarkOfficialRuntimeConfig,
    _load_revmark_models,
)


@pytest.mark.quick
def test_revmark_model_loader_uses_source_cwd_for_official_relative_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REVMark 官方 TAsBlock 使用相对路径读取 ME_Spynet_Full.pth, loader 必须临时切换 cwd。"""

    source_dir = tmp_path / "revmark_source"
    checkpoint_dir = source_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (source_dir / "ME_Spynet_Full.pth").write_bytes(b"motion")
    (checkpoint_dir / "Encoder.pth").write_bytes(b"encoder")
    (checkpoint_dir / "Decoder.pth").write_bytes(b"decoder")
    (source_dir / "REVMark.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "class _Model:",
                "    def __init__(self, *args, **kwargs):",
                "        if not Path('ME_Spynet_Full.pth').exists():",
                "            raise FileNotFoundError('ME_Spynet_Full.pth')",
                "        self.tasblock = type('TasBlock', (), {'enable': False})()",
                "    def to(self, device):",
                "        self.device = device",
                "        return self",
                "    def eval(self):",
                "        return self",
                "    def load_state_dict(self, state):",
                "        self.state = state",
                "        return None",
                "class Encoder(_Model):",
                "    pass",
                "class Decoder(_Model):",
                "    pass",
                "def framenorm(batch):",
                "    return batch",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sys.modules.pop("REVMark", None)
    import torch

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    cwd_before = Path.cwd()
    encoder, decoder, _, device = _load_revmark_models(
        REVMarkOfficialRuntimeConfig(
            run_root=str(tmp_path / "run"),
            bundle_root=str(tmp_path / "bundle"),
            source_dir=str(source_dir),
            repo_root=str(tmp_path),
            resource_root=str(tmp_path / "resources"),
            device="cpu",
        )
    )

    assert Path.cwd() == cwd_before
    assert device == "cpu"
    assert encoder.tasblock.enable is True
    assert decoder.tasblock.enable is True
    sys.modules.pop("REVMark", None)
