"""验证正式生成模型注册、不可变 revision 与 phase 输入契约。"""

from __future__ import annotations

from types import SimpleNamespace
import tomllib

import pytest

from experiments.generative_video_model_probe.colab_runtime import (
    LTX_VIDEO_CROSS_MODEL_ID,
    WAN21_PRIMARY_MODEL_ID,
    _model_family_from_id,
    _resolve_generation_model_commit,
    _scheduler_id_for_model,
    _trajectory_time_grid_id_for_model,
    validate_generation_model_provenance,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _invoke_pipeline_loader,
)
from main.methods.state_space_watermark.formal_detector import (
    REQUIRED_FLOW_PHASE_OBSERVATION_FIELDS,
    _observation_from_mapping,
)


def _complete_phase(value: float = 0.0) -> dict[str, float]:
    """构造全部字段均显式存在的最小 phase, 用于区分0与缺失值。"""

    return {
        field_name: value
        for field_name in REQUIRED_FLOW_PHASE_OBSERVATION_FIELDS
    }


@pytest.mark.quick
def test_generation_model_registry_rejects_unknown_ltx_like_identifier() -> None:
    """包含 `ltx` 字样的未知 ID 不得再静默进入 LTX pipeline。"""

    assert _model_family_from_id(WAN21_PRIMARY_MODEL_ID) == (
        "diffusers_wan21_flow_matching_dit"
    )
    assert _model_family_from_id(LTX_VIDEO_CROSS_MODEL_ID) == (
        "diffusers_ltx_video"
    )
    assert _scheduler_id_for_model(WAN21_PRIMARY_MODEL_ID).startswith("wan21_")
    assert _trajectory_time_grid_id_for_model(LTX_VIDEO_CROSS_MODEL_ID).startswith(
        "ltx_"
    )
    with pytest.raises(ValueError, match="未注册"):
        _model_family_from_id("example/unknown-ltx-video")


@pytest.mark.quick
def test_huggingface_revision_is_resolved_to_immutable_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """branch 或 tag 必须先由 Hub 解析为 commit, 不能原样写入正式记录。"""

    import huggingface_hub

    resolved_commit = "a" * 40
    observed: dict[str, object] = {}

    def fake_model_info(model_id: str, **kwargs: object) -> object:
        observed.update(model_id=model_id, **kwargs)
        return SimpleNamespace(sha=resolved_commit)

    monkeypatch.setattr(huggingface_hub, "model_info", fake_model_info)
    commit, source = _resolve_generation_model_commit(
        WAN21_PRIMARY_MODEL_ID,
        requested_revision="paper-frozen-tag",
        hf_token="token",
    )

    assert commit == resolved_commit
    assert source == "configured_revision_huggingface_resolved_commit"
    assert observed["revision"] == "paper-frozen-tag"


@pytest.mark.quick
def test_offline_resolution_only_accepts_explicit_immutable_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """离线模式不能把可漂移 branch 伪装成不可变模型版本。"""

    import huggingface_hub

    def fail_model_info(*_args: object, **_kwargs: object) -> object:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "model_info", fail_model_info)
    configured_commit = "b" * 40
    commit, source = _resolve_generation_model_commit(
        LTX_VIDEO_CROSS_MODEL_ID,
        requested_revision=configured_commit,
        hf_token=None,
    )
    assert commit == configured_commit
    assert source == "configured_immutable_commit_offline"

    with pytest.raises(RuntimeError, match="不可变 commit"):
        _resolve_generation_model_commit(
            LTX_VIDEO_CROSS_MODEL_ID,
            requested_revision="main",
            hf_token=None,
        )


@pytest.mark.quick
def test_formal_generation_provenance_requires_registered_family_and_commit() -> None:
    """正式 record 缺少 commit 或冻结状态时必须 fail-closed。"""

    valid = {
        "generation_model_id": WAN21_PRIMARY_MODEL_ID,
        "generation_model_family": "diffusers_wan21_flow_matching_dit",
        "generation_model_commit_or_hash": "c" * 40,
        "generation_model_revision_source": (
            "huggingface_default_revision_resolved_commit"
        ),
        "generation_model_revision_resolution_status": "resolved_and_frozen",
    }
    assert validate_generation_model_provenance(valid) == "c" * 40

    with pytest.raises(ValueError, match="缺少不可变模型 commit"):
        validate_generation_model_provenance(
            {**valid, "generation_model_commit_or_hash": None}
        )
    with pytest.raises(ValueError, match="未注册"):
        validate_generation_model_provenance(
            {**valid, "generation_model_id": "example/not-registered"}
        )


@pytest.mark.quick
def test_formal_phase_missing_field_fails_but_explicit_zero_is_legal() -> None:
    """必要观测缺失不得补0, 调用方显式测得的0仍应完整保留。"""

    phase = _complete_phase(0.0)
    observation = _observation_from_mapping(phase)
    assert observation.flow_phase == 0.0
    assert observation.endpoint_score == 0.0
    assert observation.replay_reliability == 0.0

    for missing_field in REQUIRED_FLOW_PHASE_OBSERVATION_FIELDS:
        incomplete = dict(phase)
        incomplete.pop(missing_field)
        with pytest.raises(KeyError, match=missing_field):
            _observation_from_mapping(incomplete)


@pytest.mark.quick
def test_replay_pipeline_loader_receives_generation_commit() -> None:
    """真实 replay loader 必须接收生成阶段记录的同一不可变 commit。"""

    observed: dict[str, str] = {}

    def loader(model_id: str, *, revision: str) -> object:
        observed.update(model_id=model_id, revision=revision)
        return object()

    commit = "d" * 40
    _invoke_pipeline_loader(
        loader,
        model_id=WAN21_PRIMARY_MODEL_ID,
        revision=commit,
    )
    assert observed == {
        "model_id": WAN21_PRIMARY_MODEL_ID,
        "revision": commit,
    }


@pytest.mark.constraint
def test_minimal_method_runtime_extra_contains_video_decode_dependencies() -> None:
    """最小方法包依赖必须覆盖核心数值、模型加载和 endpoint 视频解码路径。"""

    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)
    base_requirements = pyproject["project"]["dependencies"]
    requirements = pyproject["project"]["optional-dependencies"]["method-runtime"]

    assert any(value.startswith("numpy>=") for value in base_requirements)
    assert any(value.startswith("torch>=") for value in requirements)
    assert any(value.startswith("diffusers>=") for value in requirements)
    assert any(value.startswith("transformers>=") for value in requirements)
    assert any(value.startswith("imageio>=") for value in requirements)
    assert any(value.startswith("imageio-ffmpeg>=") for value in requirements)
