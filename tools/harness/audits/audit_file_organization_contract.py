"""审计基础目录边界是否存在。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.json_report import build_report, exit_with_report

REQUIRED_PATHS = [
    ".codex/project_contract.md",
    "docs/file_organization.md",
    "tools/harness/run_all_audits.py",
    "tests/constraints",
]
LOCAL_RUNTIME_OUTPUT_DIRS = ["outputs"]


def _is_runtime_output_ignored(root_path: Path, relative: str) -> bool:
    """判断本地运行输出目录是否被 `.gitignore` 明确排除。

    该检查的目的不是放宽输出治理, 而是区分“本地运行输出目录”和“可能被提交的正式文件”。
    如果 `outputs/` 没有被 `.gitignore` 排除, 则仍然视为违反文件组织契约。
    """
    gitignore_path = root_path / ".gitignore"
    if not gitignore_path.exists():
        return False
    ignored_patterns = {
        line.strip().rstrip("/")
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return relative.rstrip("/") in ignored_patterns


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    for relative in REQUIRED_PATHS:
        checked_paths.append(relative)
        if not (root_path / relative).exists():
            violations.append({"path": relative, "reason": "required_path_missing"})
    for relative in LOCAL_RUNTIME_OUTPUT_DIRS:
        checked_paths.append(relative)
        output_path = root_path / relative
        if output_path.exists() and not _is_runtime_output_ignored(root_path, relative):
            violations.append({"path": relative, "reason": "runtime_output_root_not_gitignored"})
    return build_report("audit_file_organization_contract", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
