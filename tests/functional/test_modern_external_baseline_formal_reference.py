"""验证 5 个主实验 modern external baseline 的 Colab 官方参考入口。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from external_baseline.official_eval_adapters.common import (
    LEGACY_OFFICIAL_EXECUTION_SUCCESS_STATUSES,
    OFFICIAL_EXECUTION_SUCCESS_STATUSES,
    OFFICIAL_REFERENCE_BUNDLE_COMPLETE_STATUS,
    OFFICIAL_REFERENCE_FAILURE_STATUS,
    OFFICIAL_REFERENCE_INCOMPLETE_STATUS,
    build_official_reference_bundle_execution_status,
)
from paper_workflow.colab_utils.modern_external_baseline_formal_reference import (
    MODERN_EXTERNAL_BASELINE_BUILD_ORDER,
    _build_runtime_closure_blocked_reference_manifest,
    _enrich_official_bundle_payload,
    _runtime_closure_blocks_reference_attempt,
    build_default_config_from_env,
    build_official_adapter_command,
    build_official_bundle_record_path,
)


EXPECTED_BASELINE_ORDER = (
    "revmark",
    "videoseal",
    "vidsig",
    "videoshield",
    "wam_frame",
)


@pytest.mark.quick
def test_modern_external_baseline_formal_reference_order_is_explicit() -> None:
    """5 个主实验 modern baseline 的独立参考 Notebook 必须使用稳定执行顺序。"""

    assert MODERN_EXTERNAL_BASELINE_BUILD_ORDER == EXPECTED_BASELINE_ORDER


@pytest.mark.quick
def test_official_reference_bundle_status_uses_specific_complete_state() -> None:
    """新 official reference bundle 整包完成状态必须使用专用语义。"""

    assert build_official_reference_bundle_execution_status(
        generated_count=46,
        expected_count=46,
        failed_count=0,
    ) == OFFICIAL_REFERENCE_BUNDLE_COMPLETE_STATUS
    assert build_official_reference_bundle_execution_status(
        generated_count=45,
        expected_count=46,
        failed_count=0,
    ) == OFFICIAL_REFERENCE_INCOMPLETE_STATUS
    assert build_official_reference_bundle_execution_status(
        generated_count=45,
        expected_count=46,
        failed_count=1,
    ) == OFFICIAL_REFERENCE_FAILURE_STATUS
    assert OFFICIAL_REFERENCE_BUNDLE_COMPLETE_STATUS in OFFICIAL_EXECUTION_SUCCESS_STATUSES
    for legacy_status in LEGACY_OFFICIAL_EXECUTION_SUCCESS_STATUSES:
        assert legacy_status in OFFICIAL_EXECUTION_SUCCESS_STATUSES


@pytest.mark.quick
def test_official_reference_runtime_sources_do_not_emit_legacy_success_status_for_new_bundles() -> None:
    """新生成端不得再把整包成功写成 `executed`。

    该测试是轻量治理测试。它不运行重型第三方模型, 只确认 5 个主实验 baseline
    的生成端统一调用状态规范函数, 从而避免后续 Colab 重跑继续产生旧语义。
    """

    runtime_paths = [
        Path("external_baseline/official_bundle_generator.py"),
        Path("external_baseline/vidsig_official_runtime.py"),
        Path("external_baseline/videoshield_official_runtime.py"),
        Path("external_baseline/revmark_official_runtime.py"),
        Path("external_baseline/wam_frame_official_runtime.py"),
    ]
    for runtime_path in runtime_paths:
        source = runtime_path.read_text(encoding="utf-8")
        assert "build_official_reference_bundle_execution_status" in source
        assert 'else "executed"' not in source


@pytest.mark.quick
def test_modern_external_baseline_default_config_is_validation_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认配置必须面向 validation_scale, 单 baseline 入口默认只闭合 official bundle。"""

    monkeypatch.delenv("SSTW_WORKFLOW_PROFILE", raising=False)
    monkeypatch.delenv("SSTW_EXTERNAL_BASELINE_REFERENCE_MAX_RECORDS", raising=False)
    monkeypatch.delenv("SSTW_RUN_OFFICIAL_RESULT_BUNDLE_PREFLIGHT_AFTER_REFERENCE", raising=False)
    monkeypatch.delenv("SSTW_RUN_EXTERNAL_BASELINE_COMPARISON_AFTER_REFERENCE", raising=False)
    monkeypatch.delenv("SSTW_RUN_SELF_CONTAINMENT_AFTER_REFERENCE", raising=False)

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
    assert config.run_official_result_bundle_preflight is False
    assert config.run_external_baseline_comparison_after_reference is False
    assert config.run_self_containment_after_reference is False

    with pytest.raises(ValueError, match="未知 modern external baseline"):
        build_default_config_from_env("legacy_proxy", repo_root=".")


@pytest.mark.quick
def test_modern_external_baseline_followup_can_be_explicitly_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """调试兼容场景可以显式开启后续统一转写, 但默认正式入口不执行。"""

    monkeypatch.setenv("SSTW_RUN_OFFICIAL_RESULT_BUNDLE_PREFLIGHT_AFTER_REFERENCE", "true")
    monkeypatch.setenv("SSTW_RUN_EXTERNAL_BASELINE_COMPARISON_AFTER_REFERENCE", "true")
    monkeypatch.setenv("SSTW_RUN_SELF_CONTAINMENT_AFTER_REFERENCE", "true")

    config = build_default_config_from_env("wam_frame", repo_root=".")

    assert config.run_official_result_bundle_preflight is True
    assert config.run_external_baseline_comparison_after_reference is True
    assert config.run_self_containment_after_reference is True


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
def test_enrich_official_bundle_payload_requires_score_and_clean_negative(tmp_path: Path) -> None:
    """formal reference helper 必须在 bundle 层阻断缺少 clean negative 的输出。"""

    bundle_path = tmp_path / "bundles" / "videoseal" / "records" / "unit.json"
    manifest_path = tmp_path / "bundles" / "videoseal" / "official_reference_execution_manifest.json"
    bundle_path.parent.mkdir(parents=True)
    payload = {
        "external_baseline_score": 0.7,
        "raw_detector_score": 0.7,
        "score_semantics": "watermark_presence_confidence",
        "score_orientation": "higher_is_more_watermarked",
        "official_score_extraction_policy": "videoseal_official_detect_presence_confidence",
        "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
    }
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="official_result_bundle_missing_clean_negative_score"):
        _enrich_official_bundle_payload(
            bundle_path,
            manifest_path,
            "videoseal",
            {
                "prompt_id": "prompt_a",
                "seed_id": "seed_a",
                "attack_name": "video_compression_runtime",
            },
        )


@pytest.mark.quick
def test_enrich_official_bundle_payload_persists_protocol_anchor_and_manifest(tmp_path: Path) -> None:
    """formal reference helper 必须把 official bundle 规范化为后续 measured_formal 输入。"""

    bundle_path = tmp_path / "bundles" / "videoseal" / "records" / "unit.json"
    manifest_path = tmp_path / "bundles" / "videoseal" / "official_reference_execution_manifest.json"
    bundle_path.parent.mkdir(parents=True)
    payload = {
        "external_baseline_score": 0.7,
        "raw_detector_score": 0.7,
        "score_semantics": "watermark_presence_confidence",
        "score_orientation": "higher_is_more_watermarked",
        "official_score_extraction_policy": "videoseal_official_detect_presence_confidence",
        "external_baseline_clean_negative_score": 0.08,
        "external_baseline_clean_negative_score_semantics": "watermark_presence_confidence",
        "external_baseline_clean_negative_video_path": "official/videoseal/clean_negative.mp4",
    }
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    enriched = _enrich_official_bundle_payload(
        bundle_path,
        manifest_path,
        "videoseal",
        {
            "prompt_id": "prompt_a",
            "seed_id": "seed_a",
            "attack_name": "video_compression_runtime",
            "trajectory_trace_id": "trace_a",
            "source_video_path": "clean.mp4",
            "attacked_video_path": "attacked.mp4",
        },
    )
    written = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert enriched["official_result_provenance"] == "repository_generated_from_third_party_official_code"
    assert enriched["official_adapter_baseline_id"] == "videoseal"
    assert enriched["official_baseline_id"] == "videoseal"
    assert enriched["official_execution_manifest_path"] == str(manifest_path)
    assert enriched["official_reference_protocol_anchor"] == "same_prompt_seed_attack_runtime_comparison_unit"
    assert enriched["runtime_comparison_unit_id"] == written["runtime_comparison_unit_id"]
    assert written["prompt_id"] == "prompt_a"


@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
def test_main_five_baseline_wrapper_modules_are_thin_entrypoints() -> None:
    """每个 baseline wrapper 只能绑定身份并转发到共享 helper。"""

    for baseline_id in EXPECTED_BASELINE_ORDER:
        module = importlib.import_module(f"paper_workflow.colab_utils.{baseline_id}_formal_reference")
        assert module.BASELINE_ID == baseline_id
        function_name = f"run_default_{baseline_id}_formal_reference_plan"
        assert callable(getattr(module, function_name))


@pytest.mark.quick
def test_main_five_baseline_formal_reference_notebooks_call_repository_helpers() -> None:
    """独立 Notebook 必须只作为 Colab 入口, 不得直接手写正式 records。"""

    for baseline_id in EXPECTED_BASELINE_ORDER:
        notebook_path = Path("paper_workflow/colab_notebooks") / f"{baseline_id}_formal_reference_colab.ipynb"
        assert notebook_path.exists()
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        first_code_cell = next(cell for cell in notebook["cells"] if cell.get("cell_type") == "code")
        first_code_source = "".join(first_code_cell.get("source", []))

        assert "drive.mount('/content/drive')" in source
        assert first_code_source.startswith("SSTW_WORKFLOW_PROFILE_VALUE = 'probe_paper'")
        assert "SSTW_WORKFLOW_PROFILE_VALUE = globals().get('SSTW_WORKFLOW_PROFILE_VALUE', 'probe_paper')" in source
        assert "NOTEBOOK_ROLE = 'external_baseline_formal_scoring'" in source
        assert f"configs/external_baselines/requirements/{baseline_id}.txt" in source
        assert "SSTW_INSTALL_BASELINE_REQUIREMENTS" in source
        assert f"from paper_workflow.colab_utils.{baseline_id}_formal_reference import" in source
        assert f"run_default_{baseline_id}_formal_reference_plan" in source
        assert "formal_reference_decision" in source
        assert "formal_comparison_scoring_colab" in source
        assert "pytest -q" in source
        assert "tools/harness/run_all_audits.py" in source
        assert "pip install -U imageio" not in source
        assert "pip install -U -r" not in source
        assert " av torchvision " not in source
        assert "write_jsonl(" not in source
        assert "runtime_detection_records.jsonl" not in source


@pytest.mark.quick
def test_formal_reference_helper_defaults_to_bundle_only_with_opt_in_unified_scoring() -> None:
    """单 baseline helper 默认只生成 official bundle, 统一转写保留为显式 opt-in。"""

    helper_text = Path("paper_workflow/colab_utils/modern_external_baseline_formal_reference.py").read_text(encoding="utf-8")

    assert "same_prompt_seed_attack_runtime_comparison_unit" in helper_text
    assert "SSTW_DISABLE_OFFICIAL_RESULT_BUNDLE_READ" in helper_text
    assert "build_external_baseline_official_resource_bootstrap_command" in helper_text
    assert "external_baseline_official_runtime_closure_requirements" in helper_text
    assert "write_official_runtime_closure_requirements" in helper_text
    assert "build_modern_baseline_command_env" in helper_text
    assert "single_baseline_notebook_default_bundle_only_formal_comparison_scoring_performs_unified_scoring" in helper_text
    assert "external_baseline_unified_measured_formal_scoring" in helper_text
    assert "build_external_baseline_comparison_command" in helper_text
    assert "measured_formal" in helper_text


@pytest.mark.quick
def test_videoseal_auto_bundle_generator_writes_fair_comparison_fields() -> None:
    """VideoSeal 自动 bundle 生成器也必须产出公平比较所需字段。"""

    generator_text = Path("external_baseline/official_bundle_generator.py").read_text(encoding="utf-8")

    assert "external_baseline_clean_negative_score" in generator_text
    assert "external_baseline_clean_negative_score_semantics" in generator_text
    assert "external_baseline_clean_negative_video_path" in generator_text
    assert "official_score_extraction_policy" in generator_text
    assert "videoseal_official_detect_presence_confidence" in generator_text
    assert "official_adapter_baseline_id" in generator_text
    assert "official_baseline_id" in generator_text
    assert "official_reference_protocol_anchor" in generator_text
    assert "same_prompt_seed_attack_runtime_comparison_unit" in generator_text
    assert "repository_generated_from_third_party_official_code" in generator_text
    assert "official_execution_manifest_path" in generator_text
    assert "official_score_formal_comparison_summary" in generator_text
    assert "attacked_for_detection = attacked_uint8.float().to(device) / 255.0" in generator_text
    assert "clean_negative_for_detection = clean_negative_uint8.float().to(device) / 255.0" in generator_text
    assert "videoseal_attacked_video_empty_after_reencode" in generator_text
