"""验证论文附件抽离契约。"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.extract_minimal_paper_package import extract_profile
from tools.harness.audits.audit_dependency_boundaries import run_audit as run_dependency_boundary_audit
from tools.harness.audits.audit_release_extraction_contract import run_audit as run_release_extraction_audit


@pytest.mark.constraint
def test_core_detector_does_not_interpret_formal_record_fields() -> None:
    """核心检测器只能消费观测、二元标签和簇标识, 不能理解论文记录角色。"""

    source_path = Path("main/methods/state_space_watermark/formal_detector.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    exact_string_constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    outer_record_fields_and_values = {
        "split",
        "sample_role",
        "attacked_positive",
        "clean_negative",
        "controlled_negative",
        "metric_status",
        "threshold_source_split",
        "test_time_threshold_update_blocked",
    }

    assert exact_string_constants.isdisjoint(outer_record_fields_and_values)


@pytest.mark.constraint
def test_dependency_boundaries_pass_for_template() -> None:
    """模板自身必须保持核心方法层可抽离。"""
    report = run_dependency_boundary_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.constraint
def test_release_extraction_contract_pass_for_template() -> None:
    """模板必须提供最小论文附件抽离规则。"""
    report = run_release_extraction_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.constraint
def test_minimal_method_package_dry_run_excludes_governance_layer(tmp_path: Path) -> None:
    """最小方法包抽离清单不得包含外层治理目录。"""
    manifest = extract_profile(Path.cwd(), tmp_path / "minimal_method_package", "minimal_method_package", dry_run=True)
    copied_files = manifest["copied_files"]
    assert copied_files
    assert all(not path.startswith(".codex/") for path in copied_files)
    assert all(not path.startswith("tools/") for path in copied_files)
    assert all(not path.startswith("tests/") for path in copied_files)
    assert all(not path.startswith("experiments/") for path in copied_files)
    assert all(not path.startswith("paper_workflow/") for path in copied_files)
    assert "configs/methods/sstw_core_method.json" in copied_files
    assert "README.md" not in copied_files
    assert "main/methods/README.md" in copied_files
    assert all(not path.startswith("configs/protocol/") for path in copied_files)
    assert all("baseline" not in path for path in copied_files)


@pytest.mark.constraint
def test_core_method_layer_does_not_recognize_experiment_variant_names() -> None:
    """核心方法只能接收参数化机制, 不得按 baseline 或消融名称切换行为。"""

    forbidden_tokens = {
        "method_variant",
        "FORMAL_METHOD_VARIANTS",
        "endpoint_only_control",
        "trajectory_only_score",
        "without_velocity_constraint",
        "without_endpoint_aware_control",
        "without_replay_uncertainty_weighting",
        "without_flow_state_admissibility",
        "generic_ssm_baseline",
    }
    for source_path in Path("main").rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert not [token for token in forbidden_tokens if token in source], (
            source_path.as_posix()
        )


@pytest.mark.constraint
def test_core_method_config_excludes_paper_experiment_protocol() -> None:
    """最小方法配置不得携带 claim、attack、baseline 或 ablation 协议。"""

    payload = json.loads(
        Path("configs/methods/sstw_core_method.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "required_claims",
        "formal_method_variants",
        "shared_attack_protocol",
        "baseline",
        "ablation",
        "probe_paper",
        "pilot_paper",
        "full_paper",
    ):
        assert forbidden not in serialized


@pytest.mark.quick
def test_extracted_minimal_method_package_is_importable_and_protocol_free(
    tmp_path: Path,
) -> None:
    """真实最小包必须可导入, 且不得携带论文实验协议文本。"""

    package_root = tmp_path / "minimal"
    manifest = extract_profile(
        Path.cwd(),
        package_root,
        "minimal_method_package",
        dry_run=False,
    )
    assert manifest["missing_paths"] == []
    forbidden_text = (
        "formal_method_variants",
        "shared_attack_protocol",
        "generic_ssm_baseline",
        "probe_paper",
        "pilot_paper",
        "full_paper",
    )
    for relative_path in manifest["copied_files"]:
        path = package_root / relative_path
        if path.suffix not in {".py", ".md", ".json", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not [token for token in forbidden_text if token in text], relative_path

    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from main.methods.state_space_watermark.formal_detector "
                "import FlowDetectorMechanismConfig; "
                "assert FlowDetectorMechanismConfig().enforce_state_admissibility"
            ),
        ],
        cwd=package_root,
        env=_isolated_python_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr


@pytest.mark.constraint
def test_paper_artifact_rebuild_package_keeps_server_runtime_without_development_layers(tmp_path: Path) -> None:
    """服务器重建包保留 workflow 配置, 但排除 Notebook 与开发治理层。"""

    manifest = extract_profile(
        Path.cwd(),
        tmp_path / "paper_artifact_rebuild_package",
        "paper_artifact_rebuild_package",
        dry_run=True,
    )
    copied_files = manifest["copied_files"]
    assert "configs/paper_workflow/generative_video_notebook_workflows.json" in copied_files
    assert all(not path.startswith("paper_workflow/") for path in copied_files)
    assert all(not path.startswith("tools/") for path in copied_files)
    assert all(not path.startswith("tests/") for path in copied_files)


def _isolated_python_environment() -> dict[str, str]:
    """构造不继承当前仓库 PYTHONPATH 的子进程环境。"""

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _assert_stage_command_target_exists(package_root: Path, command: list[str]) -> None:
    """验证 dry-run 中的 Python 模块或脚本目标真实存在于抽离包。"""

    assert command, "stage command 不能为空"
    if len(command) >= 3 and command[1] == "-m":
        module_relative = Path(*str(command[2]).split("."))
        module_file = package_root / module_relative.with_suffix(".py")
        package_file = package_root / module_relative / "__init__.py"
        assert module_file.is_file() or package_file.is_file(), command
        return
    if len(command) >= 2 and str(command[1]).endswith(".py"):
        assert (package_root / str(command[1])).is_file(), command


@pytest.mark.quick
def test_extracted_paper_artifact_rebuild_package_is_self_contained(tmp_path: Path) -> None:
    """真实抽离后验证 imports、服务器 dry-run 和全部 stage 命令目标。"""

    # Windows 的传统路径长度上限较低, 使用短目录名可让真实抽离覆盖深层 baseline 源码。
    package_root = tmp_path / "p"
    manifest = extract_profile(
        Path.cwd(),
        package_root,
        "paper_artifact_rebuild_package",
        dry_run=False,
    )
    assert manifest["missing_paths"] == []
    assert manifest["package_execution_mode"] == "paper_artifact_rebuild_package"
    assert manifest["development_checks_packaged"] is False
    assert not (package_root / "paper_workflow").exists()
    assert not (package_root / "tools").exists()
    assert not (package_root / "tests").exists()

    import_program = """
import importlib
import json
from pathlib import Path

names = [
    "main.methods.state_space_watermark.formal_detector",
    "runtime.core.progress",
    "evaluation.protocol.paper_profile_contract",
    "external_baseline.baseline_registry",
    "experiments.generative_video_model_probe.paper_result_artifact_builders",
    "workflows.generative_video_paper",
    "scripts.run_generative_video_server_workflow",
]
print(json.dumps({name: str(Path(importlib.import_module(name).__file__).resolve()) for name in names}))
"""
    imported = subprocess.run(
        [sys.executable, "-c", import_program],
        cwd=package_root,
        env=_isolated_python_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    imported_paths = json.loads(imported.stdout)
    assert set(imported_paths)
    for imported_path in imported_paths.values():
        Path(imported_path).resolve().relative_to(package_root.resolve())

    runtime_root = tmp_path / "server_runtime"
    dry_run = subprocess.run(
        [
            sys.executable,
            "scripts/run_generative_video_server_workflow.py",
            "--project-root",
            str(runtime_root),
            "--repo-root",
            str(package_root),
            "--workflow-profile",
            "probe_paper",
            "--pipeline",
            "paper_protocol_complete",
            "--dry-run",
        ],
        cwd=package_root,
        env=_isolated_python_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    decision = json.loads(dry_run.stdout)
    assert decision["server_workflow_decision"] == "DRY_RUN"
    assert decision["package_execution_mode"] == "paper_artifact_rebuild_package"

    skipped_development_checks = 0
    command_count = 0
    for role in decision["pipeline_results"]:
        for stage in role.get("stage_plan", []):
            if stage.get("stage_name") == "quick_tests_and_harness":
                assert stage["stage_execution_status"] == "skipped_in_extracted_package"
                skipped_development_checks += 1
            command = stage.get("command")
            if isinstance(command, list):
                command_count += 1
                _assert_stage_command_target_exists(package_root, [str(item) for item in command])
    assert skipped_development_checks > 0
    assert command_count > 0
