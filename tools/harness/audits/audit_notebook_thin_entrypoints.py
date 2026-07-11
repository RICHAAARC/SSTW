"""审计全部 Colab Notebook 是否保持为通用薄入口。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.json_report import build_report, exit_with_report


NOTEBOOK_ROOT = Path("paper_workflow/colab_notebooks")
WORKFLOW_CONFIG_PATH = Path("configs/paper_workflow/generative_video_notebook_workflows.json")
DELEGATION_TOKENS = (
    "from workflows",
    "import workflows",
    "from paper_workflow.colab_utils",
    "from paper_workflow.notebook_utils",
)
FORBIDDEN_WRITER_TOKENS = (
    "write_json(",
    "write_jsonl(",
    "write_csv(",
    ".write_text(",
    ".write_bytes(",
    "json.dump(",
    "csv.writer(",
    "np.save(",
    "torch.save(",
)
DIRECT_INNER_IMPORT_PATTERN = re.compile(
    r"(?m)^\s*(?:from|import)\s+(?:main|runtime|evaluation|experiments|external_baseline)(?:\.|\s|$)"
)
LOCAL_IMPLEMENTATION_PATTERN = re.compile(r"(?m)^\s*(?:async\s+def|def|class)\s+[A-Za-z_]\w*")


def _read_notebook(path: Path) -> dict[str, Any]:
    """读取 Notebook JSON 并验证顶层类型。"""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Notebook 顶层必须是对象: {path}")
    return payload


def _code_cells(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """返回 Notebook 中的代码单元。"""

    return [
        cell
        for cell in payload.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]


def _cell_source(cell: dict[str, Any]) -> str:
    """把 Notebook cell source 规范化为连续文本。"""

    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(item) for item in source)
    return ""


def _configured_notebook_paths(root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """读取统一 workflow 配置中登记的 Notebook 路径。"""

    config_path = root / WORKFLOW_CONFIG_PATH
    if not config_path.is_file():
        return [], [{"path": WORKFLOW_CONFIG_PATH.as_posix(), "reason": "missing_workflow_config"}]
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return [], [{
            "path": WORKFLOW_CONFIG_PATH.as_posix(),
            "reason": "invalid_workflow_config_json",
            "detail": str(exc),
        }]
    paths: list[str] = []
    for role_name, role in dict(config.get("notebook_roles") or {}).items():
        if not isinstance(role, dict):
            continue
        primary = str(role.get("notebook_path") or "").strip()
        if primary:
            paths.append(primary)
        examples = role.get("notebook_path_examples") or []
        if isinstance(examples, list):
            paths.extend(str(item) for item in examples if str(item))
    return sorted(set(paths)), []


def run_audit(root: str | Path) -> dict[str, Any]:
    """检查 Notebook 只承担挂载、鉴权、参数选择和 workflow 调用。"""

    root_path = Path(root)
    notebook_root = root_path / NOTEBOOK_ROOT
    checked_paths = [WORKFLOW_CONFIG_PATH.as_posix()]
    violations: list[dict[str, Any]] = []
    configured_paths, config_violations = _configured_notebook_paths(root_path)
    violations.extend(config_violations)
    for relative_path in configured_paths:
        if not (root_path / relative_path).is_file():
            violations.append({
                "path": relative_path,
                "reason": "configured_notebook_missing",
            })

    notebook_paths = sorted(notebook_root.glob("*.ipynb")) if notebook_root.is_dir() else []
    if not notebook_paths:
        violations.append({
            "path": NOTEBOOK_ROOT.as_posix(),
            "reason": "notebook_entrypoints_missing",
        })
    for notebook_path in notebook_paths:
        relative = notebook_path.relative_to(root_path).as_posix()
        checked_paths.append(relative)
        try:
            payload = _read_notebook(notebook_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            violations.append({
                "path": relative,
                "reason": "invalid_notebook_json",
                "detail": str(exc),
            })
            continue
        cells = _code_cells(payload)
        source = "\n".join(_cell_source(cell) for cell in cells)
        if not cells:
            violations.append({"path": relative, "reason": "notebook_has_no_code_cells"})
        if not any(token in source for token in DELEGATION_TOKENS):
            violations.append({
                "path": relative,
                "reason": "notebook_missing_repository_workflow_delegation",
            })
        if LOCAL_IMPLEMENTATION_PATTERN.search(source):
            violations.append({
                "path": relative,
                "reason": "notebook_contains_local_function_or_class_implementation",
            })
        if DIRECT_INNER_IMPORT_PATTERN.search(source):
            violations.append({
                "path": relative,
                "reason": "notebook_imports_inner_method_or_evaluation_layer_directly",
            })
        for token in FORBIDDEN_WRITER_TOKENS:
            if token in source:
                violations.append({
                    "path": relative,
                    "reason": "notebook_contains_direct_formal_artifact_writer",
                    "token": token,
                })
        if any(cell.get("outputs") for cell in cells):
            violations.append({"path": relative, "reason": "notebook_contains_checked_in_outputs"})
        if any(cell.get("execution_count") is not None for cell in cells):
            violations.append({"path": relative, "reason": "notebook_contains_execution_counts"})

    return build_report(
        "audit_notebook_thin_entrypoints",
        "fail" if violations else "pass",
        violations,
        checked_paths,
    )


def main() -> None:
    """运行 Notebook 薄入口审计并输出规范报告。"""

    exit_with_report(run_audit(ROOT))


if __name__ == "__main__":
    main()
