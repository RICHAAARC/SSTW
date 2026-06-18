"""检查 small-scale claim pilot gate 结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.generative_video_model_probe.pilot_claim_gate import build_small_scale_claim_pilot_audit, write_small_scale_claim_pilot_audit


def check_small_scale_claim_pilot_results(run_root: str | Path, write_outputs: bool = False) -> dict:
    """检查 small-scale claim pilot 是否满足进入 full experiment 的条件。"""
    return write_small_scale_claim_pilot_audit(run_root) if write_outputs else build_small_scale_claim_pilot_audit(run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 small-scale claim pilot gate 结果。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--write-outputs", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    payload = check_small_scale_claim_pilot_results(args.run_root, write_outputs=args.write_outputs)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
