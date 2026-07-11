"""按抽离 profile 生成论文附件候选目录。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExtractionProfile:
    """表示一个可执行的论文附件抽离 profile。"""

    profile_name: str
    include_paths: tuple[str, ...]
    exclude_parts: tuple[str, ...]


PROFILES = {
    "minimal_method_package": ExtractionProfile(
        profile_name="minimal_method_package",
        include_paths=(
            "main/methods",
            "configs/methods",
            "pyproject.toml",
        ),
        exclude_parts=(
            ".codex",
            "tools",
            "tests",
            "experiments",
            "scripts",
            "requirements",
            "/paper_workflow",
            "audit_reports",
            "outputs",
            "output",
            "result",
            "results",
            "logs",
            "__pycache__",
            ".pytest_cache",
        ),
    ),
    "paper_artifact_rebuild_package": ExtractionProfile(
        profile_name="paper_artifact_rebuild_package",
        include_paths=(
            "main",
            "runtime",
            "evaluation",
            "external_baseline",
            "configs",
            "experiments",
            "workflows",
            "scripts",
            "requirements",
            "docs/artifact_rebuild.md",
            "docs/field_registry.md",
            "docs/file_organization.md",
            "docs/release_boundary.md",
            "docs/extraction_profiles.md",
            "docs/intermediate_state_governance.md",
            "README.md",
            "pyproject.toml",
        ),
        exclude_parts=(
            ".codex",
            "tools",
            "tests",
            "/paper_workflow",
            "audit_reports",
            "outputs",
            "output",
            "result",
            "results",
            "logs",
            "__pycache__",
            ".pytest_cache",
        ),
    ),
}


def _source_git_provenance(root_path: Path) -> tuple[str | None, bool]:
    """读取抽离源仓库的 commit 与干净状态, 供服务器预检验证。"""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root_path,
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root_path,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, False
    commit_text = commit.stdout.strip() if commit.returncode == 0 else ""
    return (commit_text or None), bool(status.returncode == 0 and status.stdout.strip() == "")


def _runtime_lock_digest(root_path: Path) -> str | None:
    """返回随服务器重建包复制的运行环境锁摘要。"""

    lock_path = root_path / "requirements" / "paper_runtime_environment_lock.json"
    if not lock_path.is_file():
        return None
    return sha256(lock_path.read_bytes()).hexdigest()


def should_skip(relative_path: Path, exclude_parts: Iterable[str]) -> bool:
    """判断相对路径是否应从抽离包中排除。"""
    normalized = relative_path.as_posix()
    parts = set(relative_path.parts)
    for excluded in exclude_parts:
        root_only = excluded.startswith("/")
        excluded_normalized = excluded.strip("/").replace("\\", "/")
        if root_only:
            if normalized == excluded_normalized or normalized.startswith(f"{excluded_normalized}/"):
                return True
            continue
        if excluded_normalized in parts or normalized == excluded_normalized or normalized.startswith(f"{excluded_normalized}/"):
            return True
    return False


def iter_copy_candidates(root_path: Path, include_path: str, profile: ExtractionProfile) -> Iterable[Path]:
    """遍历某个 include path 下允许复制的文件。"""
    source = root_path / include_path
    if not source.exists():
        return
    if source.is_file():
        relative = source.relative_to(root_path)
        if not should_skip(relative, profile.exclude_parts):
            yield source
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root_path)
        if should_skip(relative, profile.exclude_parts):
            continue
        yield path


def extract_profile(root: str | Path, output: str | Path, profile_name: str, dry_run: bool = False) -> dict:
    """按指定 profile 复制文件, 并返回抽离清单。"""
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    if profile_name not in PROFILES:
        raise ValueError(f"不支持的抽离 profile: {profile_name}")
    profile = PROFILES[profile_name]
    source_git_commit, source_git_tree_clean = _source_git_provenance(root_path)

    copied_files: list[str] = []
    missing_paths: list[str] = []
    for include_path in profile.include_paths:
        source = root_path / include_path
        if not source.exists():
            missing_paths.append(include_path)
            continue
        for source_file in iter_copy_candidates(root_path, include_path, profile):
            relative = source_file.relative_to(root_path)
            copied_files.append(relative.as_posix())
            if dry_run:
                continue
            target_file = output_path / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)

    manifest = {
        "profile_name": profile.profile_name,
        "package_execution_mode": profile.profile_name,
        "root_path": str(root_path),
        "output_path": str(output_path),
        "copied_files": sorted(copied_files),
        "missing_paths": missing_paths,
        "excluded_parts": list(profile.exclude_parts),
        "dry_run": dry_run,
        "development_checks_packaged": False,
        "development_checks_execution_policy": "run_in_development_repository_before_extraction",
        "source_git_commit": source_git_commit,
        "source_git_tree_clean": source_git_tree_clean,
    }
    if profile.profile_name == "paper_artifact_rebuild_package":
        manifest["paper_runtime_environment_lock_sha256"] = _runtime_lock_digest(
            root_path
        )
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "extraction_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="按治理 profile 抽离论文附件候选目录。")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="minimal_method_package",
        help="选择抽离 profile。",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="仓库根目录。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出目录。建议使用 release_packages/ 下的未提交目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出将要复制的文件清单, 不写入文件。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    manifest = extract_profile(args.root, args.output, args.profile, args.dry_run)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
