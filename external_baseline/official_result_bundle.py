"""现代外部 baseline 项目内 official bundle cache 完整性检查。

该模块属于 external_baseline 工程边界。它不生成第三方分数, 只检查
Google Drive 或本地目录中是否已经存在由本项目 workflow 调用第三方官方代码、
官方 API 或官方原生命令生成的结果 JSON。这样可以把“缺权重 / 缺 checkpoint /
缺项目内 official bundle cache”的阻断提前暴露, 避免 validation-scale 正式门禁
在长时间 GPU 运行结束后才失败。
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
import json
import os
from pathlib import Path
from typing import Any

from external_baseline.official_eval_adapters.common import (
    official_bundle_candidate_paths,
    official_result_bundle_roots,
    read_json,
    validate_clean_negative_payload,
    validate_complete_official_bundle_baseline_identity,
    validate_repository_generated_bundle,
    validate_score_payload,
)
from external_baseline.score_semantics import validate_official_score_extraction_payload
from external_baseline.runtime_trace_io import comparable_detection_records


MODERN_BASELINE_IDS = (
    "videoshield",
    "sigmark",
    "videomark",
    "vidsig",
    "videoseal",
)


def _path_env_exists(env_name: str) -> bool:
    """检查路径型环境变量是否存在且指向已落盘文件或目录。"""
    value = os.environ.get(env_name, "").strip()
    return bool(value) and Path(value).expanduser().exists()


def _baseline_runtime_resource_ready(baseline_id: str) -> tuple[bool, str]:
    """检查当前 Colab 会话是否具备直接运行官方 adapter 的关键资源。

    该检查是保守的。返回 True 只表示“有机会直接运行”, 不代表第三方依赖已经
    完整安装。真正分数仍由 official adapter 执行时 fail closed。
    """
    upper = baseline_id.upper()
    if os.environ.get(f"SSTW_{upper}_NATIVE_EVAL_COMMAND", "").strip():
        return True, "native_official_command_configured"
    if baseline_id == "videoshield" and _path_env_exists("SSTW_VIDEOSHIELD_RESULT_JSON"):
        return True, "videoshield_official_result_json_configured"
    if baseline_id == "sigmark" and _path_env_exists("SSTW_SIGMARK_BIT_ACCURACY_NPZ"):
        if _path_env_exists("SSTW_SIGMARK_CLEAN_NEGATIVE_BIT_ACCURACY_NPZ"):
            return True, "sigmark_official_npz_and_clean_negative_npz_configured"
        return False, "sigmark_clean_negative_npz_required_for_fair_calibration"
    if baseline_id == "videomark" and _path_env_exists("SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON"):
        if _path_env_exists("SSTW_VIDEOMARK_CLEAN_NEGATIVE_RESULTS_JSON"):
            return True, "videomark_temporal_results_and_clean_negative_results_configured"
        return False, "videomark_clean_negative_results_required_for_fair_calibration"
    if baseline_id == "vidsig":
        return False, "vidsig_requires_project_owned_generate_ms_official_bundle_or_native_command"
    if baseline_id == "videoseal":
        return True, "videoseal_repository_api_runtime_attempt_allowed"
    return False, "missing_runtime_resource_or_bundle"


def _as_args(record: dict[str, Any]) -> SimpleNamespace:
    """把 runtime detection record 转换成 bundle 路径解析需要的参数对象。"""
    return SimpleNamespace(
        attack_name=str(record.get("attack_name") or ""),
        prompt_id=str(record.get("prompt_id") or ""),
        seed_id=str(record.get("seed_id") or ""),
        trajectory_trace_id=str(record.get("trajectory_trace_id") or ""),
    )


def _find_valid_bundle_path(baseline_id: str, record: dict[str, Any]) -> tuple[Path | None, str | None]:
    """查找并校验单条 comparison unit 的项目内 official bundle JSON。

    找到旧格式或外部补交 bundle 时不直接中断整个 preflight, 而是把该条视为无效
    bundle 并返回原因。这样 Colab 仍能写出 fail-closed artifact, 便于用户修复。
    """

    last_invalid_reason: str | None = None
    for candidate in official_bundle_candidate_paths(baseline_id=baseline_id, args=_as_args(record)):
        if not candidate.exists():
            continue
        try:
            payload = read_json(candidate)
            validate_score_payload(payload)
            validate_repository_generated_bundle(payload, candidate, baseline_id=baseline_id)
            validate_complete_official_bundle_baseline_identity(payload, candidate, baseline_id=baseline_id)
            validate_clean_negative_payload(payload)
            validate_official_score_extraction_payload(payload)
        except Exception as exc:
            last_invalid_reason = f"{candidate}:{exc}"
            continue
        return candidate, None
    return None, last_invalid_reason


def build_official_result_bundle_preflight(
    run_root: str | Path,
    baseline_ids: tuple[str, ...] = MODERN_BASELINE_IDS,
) -> dict[str, Any]:
    """构建现代外部 baseline 官方结果包完整性审计。"""
    run_root = Path(run_root)
    detection_records = comparable_detection_records(run_root)
    roots = official_result_bundle_roots()
    rows: list[dict[str, Any]] = []
    missing_examples: list[dict[str, Any]] = []
    present_count = 0
    expected_count = 0
    runtime_ready_baselines: list[str] = []
    bundle_ready_baselines: set[str] = set()

    for baseline_id in baseline_ids:
        runtime_ready, runtime_mode = _baseline_runtime_resource_ready(baseline_id)
        if runtime_ready:
            runtime_ready_baselines.append(baseline_id)
        baseline_present = 0
        baseline_expected = len(detection_records)
        for record in detection_records:
            expected_count += 1
            bundle_path, invalid_reason = _find_valid_bundle_path(baseline_id, record)
            if bundle_path is not None:
                present_count += 1
                baseline_present += 1
                bundle_ready_baselines.add(baseline_id)
            elif not runtime_ready and len(missing_examples) < 20:
                missing_examples.append({
                    "baseline_id": baseline_id,
                    "prompt_id": record.get("prompt_id"),
                    "seed_id": record.get("seed_id"),
                    "attack_name": record.get("attack_name"),
                    "trajectory_trace_id": record.get("trajectory_trace_id"),
                    "candidate_paths": [
                        str(path) for path in official_bundle_candidate_paths(
                            baseline_id=baseline_id,
                            args=_as_args(record),
                        )
                    ],
                    "invalid_bundle_reason": invalid_reason,
                })
        rows.append({
            "baseline_id": baseline_id,
            "runtime_resource_ready": runtime_ready,
            "runtime_resource_mode": runtime_mode,
            "bundle_expected_count": baseline_expected,
            "bundle_present_count": baseline_present,
            "bundle_missing_count": max(0, baseline_expected - baseline_present),
            "bundle_complete": bool(detection_records) and baseline_present == baseline_expected,
            "strict_resource_ready": runtime_ready or (bool(detection_records) and baseline_present == baseline_expected),
        })

    strict_ready_rows = [row for row in rows if row["strict_resource_ready"]]
    missing_rows = [row for row in rows if not row["strict_resource_ready"]]
    decision = "PASS" if detection_records and not missing_rows else "FAIL"
    return {
        "artifact_name": "external_baseline_official_result_bundle_preflight_decision.json",
        "manifest_kind": "external_baseline_official_result_bundle_preflight",
        "run_root": str(run_root),
        "official_result_bundle_preflight_decision": decision,
        "claim_support_status": "official_baseline_resources_ready_for_strict_gate" if decision == "PASS" else "official_baseline_resources_blocked",
        "official_result_bundle_roots": [str(root) for root in roots],
        "official_result_bundle_root_count": len(roots),
        "comparable_detection_record_count": len(detection_records),
        "baseline_count": len(baseline_ids),
        "expected_bundle_result_count": expected_count,
        "present_bundle_result_count": present_count,
        "missing_bundle_result_count": max(0, expected_count - present_count),
        "runtime_ready_baselines": runtime_ready_baselines,
        "bundle_ready_baselines": sorted(bundle_ready_baselines),
        "strict_ready_baseline_count": len(strict_ready_rows),
        "strict_missing_baseline_count": len(missing_rows),
        "strict_missing_baselines": [row["baseline_id"] for row in missing_rows],
        "baseline_resource_rows": rows,
        "missing_bundle_examples": missing_examples,
    }


def write_official_result_bundle_preflight(
    run_root: str | Path,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """写出官方结果包 preflight artifact。"""
    audit = build_official_result_bundle_preflight(run_root)
    output_path = Path(output_json) if output_json else Path(run_root) / "artifacts" / "external_baseline_official_result_bundle_preflight_decision.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="检查现代外部 baseline 官方结果包是否足以通过严格门禁。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    payload = write_official_result_bundle_preflight(args.run_root, args.output_json or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
