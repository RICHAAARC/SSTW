"""现代视频水印官方命令桥接器。

该模块不实现任何第三方 baseline 算法本体。它只把用户在 Colab 中安装好的官方实现、
官方权重和官方 detector / extractor 命令桥接到 SSTW 统一 JSON 输出契约。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping


SCORE_FIELDS = (
    "external_baseline_score",
    "watermark_score",
    "detection_score",
    "score",
    "bit_accuracy",
    "confidence",
)


def official_eval_command_env_var_for(baseline_id: str) -> str:
    """根据 baseline_id 推导官方原生命令环境变量名。"""
    return f"SSTW_{baseline_id.upper()}_OFFICIAL_EVAL_COMMAND"


def _safe_float(value: Any) -> float:
    """把官方输出中的常见数值字段转为 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_score(payload: Mapping[str, Any]) -> float:
    """从官方输出 JSON 中提取统一 baseline score。"""
    for field in SCORE_FIELDS:
        if field in payload:
            return _safe_float(payload.get(field))
    if "detected" in payload:
        return 1.0 if bool(payload.get("detected")) else 0.0
    raise ValueError("official_output_missing_score")


def _read_json(path: str | Path) -> dict[str, Any]:
    """读取官方命令输出 JSON。"""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("official_output_json_must_be_object")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写出 SSTW command adapter 可读取的标准 JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _official_result_bundle_read_enabled() -> bool:
    """判断当前 bridge 是否允许通过项目内 official bundle cache 完成读取。

    该函数不直接信任 bundle 内容。真正的 provenance、score 字段和执行
    manifest 校验仍由 `external_baseline.official_eval_adapters.common`
    完成。这里仅用于决定是否可以把“官方源码目录不存在”的检查延后到
    repository official adapter。
    """

    if os.environ.get("SSTW_DISABLE_OFFICIAL_RESULT_BUNDLE_READ", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return any([
        os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", "").strip(),
        os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS", "").strip(),
        os.environ.get("SSTW_EXTERNAL_BASELINE_BUNDLE_ROOT", "").strip(),
    ])


def _official_eval_command_uses_repository_adapter(baseline_id: str) -> bool:
    """判断内部官方命令是否是项目内 fail-closed official adapter。

    只有这种命令能够在源码目录缺失时安全地读取项目内 official bundle。
    用户自定义 native command 仍然需要 bridge 在入口处检查官方源码目录,
    防止把缺失依赖延后成难以理解的第三方报错。
    """

    template = os.environ.get(official_eval_command_env_var_for(baseline_id), "").strip()
    return "external_baseline.official_eval_adapters" in template


def _format_official_command(args: argparse.Namespace, official_output_json_path: Path) -> list[str]:
    """把用户配置的官方命令模板格式化为 argv。

    用户命令必须写入 `official_output_json_path`, 而不是直接写 SSTW 的最终
    `output_json_path`。这样桥接器可以把不同官方字段稳定归一化为统一分数字段。
    """
    env_var = official_eval_command_env_var_for(args.baseline_id)
    template = os.environ.get(env_var, "").strip()
    if not template:
        raise RuntimeError(f"missing_official_eval_command_env:{env_var}")
    values = {
        "baseline_id": args.baseline_id,
        "official_source_dir": args.official_source_dir,
        "source_video_path": args.source_video,
        "attacked_video_path": args.attacked_video,
        "attack_name": args.attack_name,
        "official_output_json_path": str(official_output_json_path),
        "raw_output_json_path": str(official_output_json_path),
        "output_json_path": str(args.output_json),
        "run_root": args.run_root or "",
        "prompt_id": args.prompt_id or "",
        "seed_id": args.seed_id or "",
        "trajectory_trace_id": args.trajectory_trace_id or "",
    }
    return shlex.split(template.format(**values), posix=os.name != "nt")


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    """执行官方命令并归一化输出。

    该函数属于通用适配层。项目特定约束是: 缺官方源码目录、缺官方命令或官方命令
    未输出 score 时必须失败, 不能用 SSTW 自身分数或视频相似度伪造外部 baseline。
    """
    official_source_dir = Path(args.official_source_dir)
    allow_repository_bundle_without_source = (
        _official_result_bundle_read_enabled()
        and _official_eval_command_uses_repository_adapter(args.baseline_id)
    )
    if not official_source_dir.is_dir() and not allow_repository_bundle_without_source:
        raise FileNotFoundError(f"official_source_dir_missing:{official_source_dir}")
    source_video = Path(args.source_video)
    attacked_video = Path(args.attacked_video)
    if not source_video.exists():
        raise FileNotFoundError(f"source_video_missing:{source_video}")
    if not attacked_video.exists():
        raise FileNotFoundError(f"attacked_video_missing:{attacked_video}")

    output_json_path = Path(args.output_json)
    official_output_json_path = output_json_path.with_name(output_json_path.stem + "_official_raw.json")
    argv = _format_official_command(args, official_output_json_path)
    timeout_sec = float(os.environ.get("SSTW_OFFICIAL_BASELINE_BRIDGE_TIMEOUT_SEC", "1800"))
    completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout_sec)
    if completed.returncode != 0:
        raise RuntimeError(f"official_eval_command_failed:{completed.returncode}:{completed.stderr[-500:]}")
    if not official_output_json_path.exists():
        raise FileNotFoundError(f"official_output_json_missing:{official_output_json_path}")

    official_payload = _read_json(official_output_json_path)
    score = round(_extract_score(official_payload), 6)
    normalized = {
        "external_baseline_score": score,
        "external_baseline_detected": official_payload.get("external_baseline_detected", official_payload.get("detected")),
        "external_baseline_bit_accuracy": official_payload.get("external_baseline_bit_accuracy", official_payload.get("bit_accuracy")),
        "external_baseline_threshold": official_payload.get("external_baseline_threshold", official_payload.get("threshold")),
        "external_baseline_distance": official_payload.get("external_baseline_distance"),
        "official_result_provenance": official_payload.get("official_result_provenance"),
        "official_result_bundle_path": official_payload.get("official_result_bundle_path"),
        "official_execution_manifest_path": official_payload.get("official_execution_manifest_path"),
        "external_baseline_source_video_path": official_payload.get(
            "external_baseline_source_video_path",
            official_payload.get("baseline_source_video_path"),
        ),
        "external_baseline_attacked_video_path": official_payload.get(
            "external_baseline_attacked_video_path",
            official_payload.get("baseline_attacked_video_path"),
        ),
        "external_baseline_generation_model_id": official_payload.get("external_baseline_generation_model_id"),
        "official_reference_protocol_anchor": official_payload.get("official_reference_protocol_anchor"),
        "runtime_comparison_unit_id": official_payload.get("runtime_comparison_unit_id"),
        "prompt_id": official_payload.get("prompt_id", args.prompt_id),
        "seed_id": official_payload.get("seed_id", args.seed_id),
        "attack_name": official_payload.get("attack_name", args.attack_name),
        "trajectory_trace_id": official_payload.get("trajectory_trace_id", args.trajectory_trace_id),
        "official_bridge_status": "measured_formal_from_official_command",
        "official_bridge_baseline_id": args.baseline_id,
        "official_bridge_raw_output_json_path": str(official_output_json_path),
        "official_bridge_command_env_var": official_eval_command_env_var_for(args.baseline_id),
        "official_bridge_stdout_tail": completed.stdout[-1000:],
        "official_bridge_stderr_tail": completed.stderr[-1000:],
    }
    _write_json(output_json_path, normalized)
    return normalized


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="SSTW 现代 baseline 官方命令桥接器。")
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--official-source-dir", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--attacked-video", required=True)
    parser.add_argument("--attack-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--prompt-id", default="")
    parser.add_argument("--seed-id", default="")
    parser.add_argument("--trajectory-trace-id", default="")
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()
    payload = run_bridge(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
