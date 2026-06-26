"""验证 6 个 modern external baseline 的 Colab 官方参考入口。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from paper_workflow.colab_utils.modern_external_baseline_formal_reference import (
    MODERN_EXTERNAL_BASELINE_BUILD_ORDER,
    build_default_config_from_env,
    build_official_adapter_command,
    build_official_bundle_record_path,
)


EXPECTED_BASELINE_ORDER = (
    "videoseal",
    "vidsig",
    "videomark",
    "videoshield",
    "spdmark",
    "sigmark",
)


@pytest.mark.quick
def test_modern_external_baseline_formal_reference_order_is_explicit() -> None:
    """6 个 modern baseline 的独立参考 Notebook 必须使用稳定执行顺序。"""

    assert MODERN_EXTERNAL_BASELINE_BUILD_ORDER == EXPECTED_BASELINE_ORDER


@pytest.mark.quick
def test_modern_external_baseline_default_config_is_validation_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认配置必须面向 validation_scale, 且未知 baseline 必须 fail closed。"""

    monkeypatch.delenv("SSTW_WORKFLOW_PROFILE", raising=False)
    monkeypatch.delenv("SSTW_EXTERNAL_BASELINE_REFERENCE_MAX_RECORDS", raising=False)

    config = build_default_config_from_env("videoseal", repo_root=".")

    assert config.baseline_id == "videoseal"
    assert config.workflow_profile == "validation_scale"
    assert config.execute_source_clone is True
    assert config.run_source_intake is True
    assert config.run_official_resource_bootstrap is True
    assert config.generate_auto_supported_bundle is True
    assert config.allow_existing_official_bundle_as_reference_input is False
    assert config.max_records is None
    assert config.run_official_runtime_closure_preflight is True
    assert config.run_official_result_bundle_preflight is True
    assert config.run_external_baseline_comparison_after_reference is True
    assert config.run_self_containment_after_reference is True

    with pytest.raises(ValueError, match="未知 modern external baseline"):
        build_default_config_from_env("legacy_proxy", repo_root=".")


@pytest.mark.quick
def test_official_bundle_record_path_sanitizes_runtime_record_tokens() -> None:
    """official bundle 路径必须能稳定映射 prompt、seed 和 attack 标识。"""

    output_path = build_official_bundle_record_path(
        Path("/bundle"),
        "videoseal",
        {
            "prompt_id": "prompt 1",
            "seed_id": "seed/2",
            "attack_name": "jpeg compression",
        },
    )

    assert output_path.as_posix().endswith("/bundle/videoseal/records/prompt_1__seed_2__jpeg_compression.json")


@pytest.mark.quick
def test_build_official_adapter_command_uses_repository_adapter_module() -> None:
    """非 VideoSeal baseline 参考运行必须通过仓库 official adapter 统一 I/O。"""

    command = build_official_adapter_command(
        baseline_id="vidsig",
        official_source_dir="/content/SSTW/external_baseline/primary/vidsig/source",
        detection_record={
            "source_video_path": "/tmp/source.mp4",
            "attacked_video_path": "/tmp/attacked.mp4",
            "attack_name": "h264",
            "run_root": "/content/drive/MyDrive/SSTW/runs/generative_video_model_probe/validation_scale",
            "prompt_id": "prompt_a",
            "seed_id": "seed_b",
            "trajectory_trace_id": "trace_c",
        },
        output_json_path="/content/drive/MyDrive/SSTW/external_baseline_official_result_bundles/validation_scale/vidsig/records/unit.json",
    )
    command_text = " ".join(command)

    assert "-m external_baseline.official_eval_adapters.vidsig" in command_text
    assert "--official-source-dir /content/SSTW/external_baseline/primary/vidsig/source" in command_text
    assert "--source-video /tmp/source.mp4" in command_text
    assert "--attacked-video /tmp/attacked.mp4" in command_text
    assert "--official-output-json /content/drive/MyDrive/SSTW/external_baseline_official_result_bundles/validation_scale/vidsig/records/unit.json" in command_text
    assert "--trajectory-trace-id trace_c" in command_text


@pytest.mark.quick
def test_six_baseline_wrapper_modules_are_thin_entrypoints() -> None:
    """每个 baseline wrapper 只能绑定身份并转发到共享 helper。"""

    for baseline_id in EXPECTED_BASELINE_ORDER:
        module = importlib.import_module(f"paper_workflow.colab_utils.{baseline_id}_formal_reference")
        assert module.BASELINE_ID == baseline_id
        function_name = f"run_default_{baseline_id}_formal_reference_plan"
        assert callable(getattr(module, function_name))


@pytest.mark.quick
def test_six_baseline_formal_reference_notebooks_call_repository_helpers() -> None:
    """独立 Notebook 必须只作为 Colab 入口, 不得直接手写正式 records。"""

    for baseline_id in EXPECTED_BASELINE_ORDER:
        notebook_path = Path("paper_workflow/colab_notebooks") / f"{baseline_id}_formal_reference_colab.ipynb"
        assert notebook_path.exists()
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        assert "drive.mount('/content/drive')" in source
        assert "SSTW_WORKFLOW_PROFILE_VALUE = 'validation_scale'" in source
        assert "NOTEBOOK_ROLE = 'external_baseline_formal_scoring'" in source
        assert f"configs/external_baselines/requirements/{baseline_id}.txt" in source
        assert "SSTW_INSTALL_BASELINE_REQUIREMENTS" in source
        assert f"from paper_workflow.colab_utils.{baseline_id}_formal_reference import" in source
        assert f"run_default_{baseline_id}_formal_reference_plan" in source
        assert "formal_reference_decision" in source
        assert "measured_formal" in source
        assert "pytest -q" in source
        assert "tools/harness/run_all_audits.py" in source
        assert "write_jsonl(" not in source
        assert "runtime_detection_records.jsonl" not in source


@pytest.mark.quick
def test_formal_reference_helper_runs_bundle_then_unified_measured_formal_scoring() -> None:
    """单 baseline helper 必须先生成 official bundle, 再调用统一 runner 转写 records。"""

    helper_text = Path("paper_workflow/colab_utils/modern_external_baseline_formal_reference.py").read_text(encoding="utf-8")

    assert "same_prompt_seed_attack_runtime_comparison_unit" in helper_text
    assert "SSTW_DISABLE_OFFICIAL_RESULT_BUNDLE_READ" in helper_text
    assert "build_external_baseline_official_resource_bootstrap_command" in helper_text
    assert "external_baseline_official_runtime_closure_requirements" in helper_text
    assert "write_official_runtime_closure_requirements" in helper_text
    assert "build_modern_baseline_command_env" in helper_text
    assert "external_baseline_unified_measured_formal_scoring" in helper_text
    assert "build_external_baseline_comparison_command" in helper_text
    assert "measured_formal" in helper_text
