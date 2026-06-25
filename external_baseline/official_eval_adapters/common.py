"""现代视频水印官方评测 adapter 的公共工具。

该模块属于通用工程层。它提供统一 CLI、官方源码检查、原生命令透传和
JSON score 校验。项目特定约束是: 任何 adapter 在无法真实调用官方实现时必须
失败, 不能回退到 SSTW 自身分数、视频相似度或随机分数。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable, Mapping, Sequence


SCORE_FIELDS = (
    "external_baseline_score",
    "watermark_score",
    "detection_score",
    "score",
    "bit_accuracy",
    "confidence",
    "detected",
)


def build_parser(description: str) -> argparse.ArgumentParser:
    """构造所有官方 adapter 共享的参数解析器。

    通用工程写法是让每个 baseline 使用同一组输入 token。项目特定要求是
    `official_output_json_path` 由 bridge 生成, wrapper 只能写这个文件, 不能直接
    修改 SSTW 聚合 records。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--official-source-dir", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--attacked-video", required=True)
    parser.add_argument("--attack-name", required=True)
    parser.add_argument("--official-output-json", required=True)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--prompt-id", default="")
    parser.add_argument("--seed-id", default="")
    parser.add_argument("--trajectory-trace-id", default="")
    return parser


def require_paths_exist(paths: Sequence[str | Path], *, label: str) -> None:
    """检查一组路径是否存在, 缺失时 fail closed。"""
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{label}_missing:{missing}")


def verify_official_source(official_source_dir: str | Path, required_files: Sequence[str]) -> Path:
    """检查官方源码目录和关键入口文件。

    该检查只证明 wrapper 面向的是第三方官方仓库结构, 不证明权重和运行环境完整。
    权重缺失必须在具体 adapter 执行时继续 fail closed。
    """
    source_dir = Path(official_source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"official_source_dir_missing:{source_dir}")
    missing = [relative for relative in required_files if not (source_dir / relative).exists()]
    if missing:
        raise FileNotFoundError(f"official_source_required_files_missing:{missing}")
    return source_dir


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 对象, 并兼容 UTF-8 BOM。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"json_payload_must_be_object:{path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写出 adapter 的官方 raw JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_path_token(value: Any) -> str:
    """把 prompt、seed、attack 等字段转换为 bundle 路径可使用的 token。

    该函数属于通用工程写法。项目特定用途是统一 Google Drive 官方 baseline
    结果包的文件命名, 防止 Colab 冷启动后因为路径命名不一致而找不到同一条
    prompt / seed / attack 的官方结果。
    """
    import re

    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def official_result_bundle_roots() -> list[Path]:
    """读取可选的官方结果包根目录列表。

    `SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS` 支持多个目录, 用
    `os.pathsep` 分隔。`SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT` 和
    历史短名 `SSTW_EXTERNAL_BASELINE_BUNDLE_ROOT` 保持兼容。
    """
    values: list[str] = []
    multi = os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS", "").strip()
    if multi:
        values.extend(item for item in multi.split(os.pathsep) if item.strip())
    single = os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", "").strip()
    if single:
        values.append(single)
    legacy = os.environ.get("SSTW_EXTERNAL_BASELINE_BUNDLE_ROOT", "").strip()
    if legacy:
        values.append(legacy)
    roots: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = Path(value).expanduser()
        key = str(path)
        if key not in seen:
            roots.append(path)
            seen.add(key)
    return roots


def official_bundle_candidate_paths(
    *,
    baseline_id: str,
    args: argparse.Namespace,
) -> list[Path]:
    """构造单条 comparison unit 对应的官方结果包候选路径。

    官方结果包用于接收那些无法在当前 Colab L4 会话中重新训练或重新生成的
    第三方 baseline 结果。它不是 proxy 分数: 每个 JSON 仍必须来自第三方官方
    代码或官方原生命令, 并保留 `official_execution_manifest_path` 或等价 provenance。
    """
    attack = _safe_path_token(args.attack_name)
    prompt = _safe_path_token(args.prompt_id)
    seed = _safe_path_token(args.seed_id)
    trace = _safe_path_token(args.trajectory_trace_id)
    candidates: list[Path] = []
    for root in official_result_bundle_roots():
        baseline_root = root / baseline_id
        candidates.extend([
            baseline_root / "records" / f"{prompt}__{seed}__{attack}.json",
            baseline_root / "records" / f"{trace}__{attack}.json",
            baseline_root / prompt / seed / f"{attack}.json",
            baseline_root / trace / f"{attack}.json",
        ])
    return candidates


def read_official_result_bundle_if_available(
    *,
    baseline_id: str,
    args: argparse.Namespace,
    source_dir: Path,
    output_json_path: Path,
) -> dict[str, Any] | None:
    """读取 Google Drive 中预先生成的官方 baseline 结果包。

    这一实现属于项目特定写法。它解决的问题是: 部分现代视频水印 baseline
    需要特定生成模型、训练出的 extractor 或高显存环境, 不能保证在同一个
    Wan2.1 Colab 会话中即时复跑。正式论文比较仍然允许读取由官方代码离线
    生成的结果包, 但必须满足两个条件:

    1. JSON 中存在可审计 score 字段。
    2. JSON 必须声明不是 SSTW proxy, 即 `official_result_provenance` 不能为
       `sstw_proxy`。
    """
    for candidate in official_bundle_candidate_paths(baseline_id=baseline_id, args=args):
        if not candidate.exists():
            continue
        payload = read_json(candidate)
        validate_score_payload(payload)
        if payload.get("official_result_provenance") == "sstw_proxy":
            raise RuntimeError(f"official_result_bundle_proxy_forbidden:{candidate}")
        return {
            **payload,
            "official_adapter_status": "measured_from_official_result_bundle",
            "official_adapter_baseline_id": baseline_id,
            "official_source_dir": str(source_dir),
            "official_result_bundle_path": str(candidate),
            "official_output_json_path": str(output_json_path),
        }
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    """把官方输出中的数值字段安全转换为 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_score(payload: Mapping[str, Any]) -> float:
    """从官方输出 JSON 中提取 SSTW 可接受的 score 字段。"""
    for field in SCORE_FIELDS:
        if field == "detected" and field in payload:
            return 1.0 if bool(payload.get(field)) else 0.0
        if field in payload:
            return safe_float(payload.get(field), 0.0)
    raise ValueError("official_output_missing_score")


def validate_score_payload(payload: Mapping[str, Any]) -> None:
    """确认输出中存在至少一个可审计分数字段。"""
    extract_score(payload)


def format_command(template: str, args: argparse.Namespace, output_json_path: Path) -> list[str]:
    """格式化用户提供的官方原生命令模板。"""
    values = {
        "official_source_dir": str(args.official_source_dir),
        "source_video_path": str(args.source_video),
        "attacked_video_path": str(args.attacked_video),
        "attack_name": str(args.attack_name),
        "official_output_json_path": str(output_json_path),
        "output_json_path": str(output_json_path),
        "run_root": str(args.run_root or ""),
        "prompt_id": str(args.prompt_id or ""),
        "seed_id": str(args.seed_id or ""),
        "trajectory_trace_id": str(args.trajectory_trace_id or ""),
    }
    return shlex.split(template.format(**values), posix=os.name != "nt")


def run_native_command_if_configured(
    *,
    baseline_id: str,
    args: argparse.Namespace,
    output_json_path: Path,
) -> dict[str, Any] | None:
    """优先执行用户提供的官方原生命令。

    每个 repository wrapper 都支持 `SSTW_<BASELINE>_NATIVE_EVAL_COMMAND`。
    这样当官方仓库提供新的正式 CLI 时, 用户不需要修改 SSTW 代码, 只需要在
    Colab 中注入原生命令即可。没有配置时返回 None, 由 adapter 的默认实现继续处理。
    """
    env_var = f"SSTW_{baseline_id.upper()}_NATIVE_EVAL_COMMAND"
    template = os.environ.get(env_var, "").strip()
    if not template:
        return None
    argv = format_command(template, args, output_json_path)
    timeout_sec = safe_float(os.environ.get("SSTW_OFFICIAL_BASELINE_NATIVE_TIMEOUT_SEC"), 3600.0)
    completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout_sec)
    if completed.returncode != 0:
        raise RuntimeError(f"native_official_command_failed:{baseline_id}:{completed.returncode}:{completed.stderr[-1000:]}")
    if not output_json_path.exists():
        raise FileNotFoundError(f"native_official_output_json_missing:{output_json_path}")
    payload = read_json(output_json_path)
    validate_score_payload(payload)
    enriched = {
        **payload,
        "official_adapter_status": "measured_by_native_official_command",
        "official_adapter_baseline_id": baseline_id,
        "official_adapter_native_command_env_var": env_var,
        "official_adapter_stdout_tail": completed.stdout[-1000:],
        "official_adapter_stderr_tail": completed.stderr[-1000:],
    }
    write_json(output_json_path, enriched)
    return enriched


def run_adapter_main(
    *,
    baseline_id: str,
    description: str,
    required_source_files: Sequence[str],
    default_runner: Callable[[argparse.Namespace, Path, Path], dict[str, Any]],
) -> None:
    """执行单个 official adapter。

    该函数先检查官方源码和输入视频, 再优先执行用户配置的官方原生命令, 最后才调用
    repository 内置 wrapper。内置 wrapper 也必须真实调用官方源码或官方模型 API。
    """
    parser = build_parser(description)
    args = parser.parse_args()
    source_dir = verify_official_source(args.official_source_dir, required_source_files)
    require_paths_exist([args.source_video, args.attacked_video], label="baseline_input_video")
    output_json_path = Path(args.official_output_json)

    payload = run_native_command_if_configured(
        baseline_id=baseline_id,
        args=args,
        output_json_path=output_json_path,
    )
    if payload is None:
        payload = default_runner(args, source_dir, output_json_path)
        validate_score_payload(payload)
        write_json(output_json_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def raise_missing_official_artifacts(baseline_id: str, details: str) -> None:
    """在官方权重或官方中间产物缺失时给出明确失败原因。"""
    raise RuntimeError(
        f"{baseline_id}_official_required_artifacts_missing:{details}. "
        "请配置对应 SSTW_<BASELINE>_NATIVE_EVAL_COMMAND, 或在 Google Drive 中提供官方权重、"
        "官方 key / message / maintained info 等可复现实验产物。"
    )
