"""生成外部 baseline source intake 治理清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_baseline.source_intake import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_REGISTRY_PATH,
    write_source_intake_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出 SSTW external baseline source intake manifest。")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_SOURCE_REGISTRY_PATH))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--execute-clone",
        action="store_true",
        help="实际执行可 clone URL 的 git clone / fetch。默认只写计划和缺口, 不访问网络。",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()
    payload = write_source_intake_artifacts(
        output_root=args.output_root,
        registry_path=args.registry_path,
        repo_root=args.repo_root,
        execute_clone=args.execute_clone,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
