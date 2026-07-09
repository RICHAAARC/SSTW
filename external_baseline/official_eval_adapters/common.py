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

from external_baseline.score_semantics import (
    extract_raw_detector_score,
    official_score_formal_comparison_summary,
    validate_official_formal_comparison_eligibility,
    validate_official_score_extraction_payload,
)


SCORE_FIELDS = (
    "raw_detector_score",
    "external_baseline_raw_detector_score",
    "detection_score",
    "confidence",
    "external_baseline_score",
    "watermark_score",
    "score",
    "bit_accuracy",
    "external_baseline_bit_accuracy",
    "confidence",
    "detected",
    "external_baseline_detected",
)
REPOSITORY_GENERATED_OFFICIAL_PROVENANCE = "repository_generated_from_third_party_official_code"
REQUIRED_COMPLETE_OFFICIAL_BASELINE_ID_FIELDS = (
    "official_adapter_baseline_id",
    "official_baseline_id",
)
OFFICIAL_REFERENCE_BUNDLE_COMPLETE_STATUS = "official_reference_bundle_complete"
OFFICIAL_REFERENCE_FAILURE_STATUS = "official_reference_failures_present"
OFFICIAL_REFERENCE_INCOMPLETE_STATUS = "bundle_record_coverage_incomplete"
LEGACY_OFFICIAL_EXECUTION_SUCCESS_STATUSES = (
    "executed",
    "completed",
    "generated",
    "ready",
)
OFFICIAL_EXECUTION_SUCCESS_STATUSES = (
    *LEGACY_OFFICIAL_EXECUTION_SUCCESS_STATUSES,
    OFFICIAL_REFERENCE_BUNDLE_COMPLETE_STATUS,
)


def build_official_reference_bundle_execution_status(
    *,
    generated_count: int,
    expected_count: int,
    failed_count: int,
) -> str:
    """统一生成 official reference bundle 整包状态。

    该函数属于项目特定写法。新生成的 formal reference bundle 必须使用
    `official_reference_bundle_complete` 表示整包完成; 旧的 `executed`、
    `completed`、`generated`、`ready` 只在读取历史 manifest 时作为兼容输入,
    不再作为新整包 manifest 的成功状态输出。
    """

    if expected_count > 0 and generated_count == expected_count and failed_count == 0:
        return OFFICIAL_REFERENCE_BUNDLE_COMPLETE_STATUS
    if failed_count:
        return OFFICIAL_REFERENCE_FAILURE_STATUS
    return OFFICIAL_REFERENCE_INCOMPLETE_STATUS


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


def resolve_existing_env_file(env_var: str) -> Path | None:
    """读取环境变量中的文件路径, 并拒绝空值、目录和不存在的路径。

    该函数属于通用防御式写法。`Path("")` 在 Python 中会解析为当前目录 `.`,
    如果只检查 `exists()` 就会把“未配置文件”误判为“当前目录存在”。对于官方
    baseline 权重、npz、json 等资源, 只有真实文件才能继续进入 adapter。
    """
    value = os.environ.get(env_var, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


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

    官方结果包用于缓存由本项目 workflow 调用第三方官方代码、官方 API 或官方
    原生命令后产生的结果。它不是外部补交分数: 每个 JSON 必须由项目内
    clone / build / run / adapt / record 链路生成, 并保留
    `official_execution_manifest_path`。
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


def _declared_official_baseline_ids(payload: Mapping[str, Any]) -> dict[str, str]:
    """提取 official bundle 中显式声明的 baseline 身份字段。

    该函数属于项目特定治理逻辑。公平比较要求每个 official bundle 只能服务于
    当前 baseline, 因此只要 payload 中任一身份字段与当前 baseline 不一致, 就
    必须 fail closed。这里同时接受历史字段 `baseline_id`, 便于旧的项目内
    manifest 在重新打包前仍能被明确审计, 但不允许多个字段互相矛盾。
    """

    declared: dict[str, str] = {}
    for field_name in ("official_adapter_baseline_id", "official_baseline_id", "baseline_id"):
        raw_value = payload.get(field_name)
        value = str(raw_value).strip() if raw_value is not None else ""
        if value:
            declared[field_name] = value
    return declared


def validate_official_bundle_baseline_identity(
    payload: Mapping[str, Any],
    candidate: str | Path,
    *,
    baseline_id: str,
) -> None:
    """校验 official bundle 声明的 baseline 身份与当前 adapter 一致。

    该函数只检查身份字段, 不检查 provenance 和执行 manifest。这样 command
    adapter 在读取官方命令输出时可以先拒绝跨 baseline 结果, 而完整 bundle
    preflight 仍由 `validate_repository_generated_bundle` 负责。
    """

    expected_baseline_id = str(baseline_id or "").strip()
    if not expected_baseline_id:
        raise RuntimeError(f"official_result_bundle_missing_expected_baseline_id:{candidate}")
    declared_ids = _declared_official_baseline_ids(payload)
    if not declared_ids:
        raise RuntimeError(
            "official_result_bundle_missing_baseline_id:"
            f"{candidate}:expected={expected_baseline_id}"
        )
    mismatched_ids = {
        field_name: value
        for field_name, value in declared_ids.items()
        if value != expected_baseline_id
    }
    if mismatched_ids:
        raise RuntimeError(
            "official_result_bundle_baseline_id_mismatch:"
            f"{candidate}:expected={expected_baseline_id}:declared={mismatched_ids}"
        )


def validate_complete_official_bundle_baseline_identity(
    payload: Mapping[str, Any],
    candidate: str | Path,
    *,
    baseline_id: str,
) -> None:
    """校验 official bundle 同时声明 adapter 身份和官方 baseline 身份。

    通用身份检查只要求至少存在一个可审计 baseline 字段。paper profile 的
    公平比较需要更强约束: 每个正式 official bundle 必须同时说明“由哪个 SSTW
    official adapter 读取”和“该 bundle 属于哪个官方 baseline”。这样后续
    self-containment 与 measured_formal 转写不会把旧式或半结构化 bundle 误认为
    完整逐 baseline 官方结果。
    """
    validate_official_bundle_baseline_identity(payload, candidate, baseline_id=baseline_id)
    expected_baseline_id = str(baseline_id or "").strip()
    missing_fields = [
        field_name
        for field_name in REQUIRED_COMPLETE_OFFICIAL_BASELINE_ID_FIELDS
        if str(payload.get(field_name) or "").strip() == ""
    ]
    if missing_fields:
        raise RuntimeError(
            "official_result_bundle_missing_complete_baseline_identity:"
            f"{candidate}:expected={expected_baseline_id}:missing={missing_fields}"
        )
    mismatched_fields = {
        field_name: str(payload.get(field_name) or "").strip()
        for field_name in REQUIRED_COMPLETE_OFFICIAL_BASELINE_ID_FIELDS
        if str(payload.get(field_name) or "").strip() != expected_baseline_id
    }
    if mismatched_fields:
        raise RuntimeError(
            "official_result_bundle_complete_baseline_identity_mismatch:"
            f"{candidate}:expected={expected_baseline_id}:declared={mismatched_fields}"
        )


def with_official_adapter_identity(
    payload: Mapping[str, Any],
    *,
    baseline_id: str,
    candidate: str | Path,
) -> dict[str, Any]:
    """给官方 adapter 输出补齐完整 baseline 身份, 并拒绝已有矛盾声明。

    该函数适用于原生命令输出和 repository adapter 输出。若输出已经声明身份,
    先检查其不与当前 baseline 冲突; 若未声明, 则由当前 adapter 统一补齐两个
    必备身份字段, 使后续 bridge 和 measured_formal 转写拥有一致的自描述输入。
    """
    declared_ids = _declared_official_baseline_ids(payload)
    if declared_ids:
        validate_official_bundle_baseline_identity(payload, candidate, baseline_id=baseline_id)
    return {
        **payload,
        "official_adapter_baseline_id": baseline_id,
        "official_baseline_id": baseline_id,
    }


def validate_repository_generated_bundle(
    payload: Mapping[str, Any],
    candidate: str | Path,
    *,
    baseline_id: str | None = None,
) -> None:
    """校验 official bundle 只能来自项目内自包含生成链路。

    该函数属于项目特定门禁: 读取 official bundle cache 是为了避免重复运行高显存
    baseline, 不是为了接受外部补交 JSON。旧的 `third_party_official_code` 只表示
    “来自第三方官方代码”, 但不能证明由本项目 workflow 生成, 因此不能继续作为
    正式 `measured_formal` 输入。
    """

    provenance = str(payload.get("official_result_provenance") or "")
    if provenance != REPOSITORY_GENERATED_OFFICIAL_PROVENANCE:
        raise RuntimeError(
            "official_result_bundle_not_repository_generated:"
            f"{candidate}:{provenance or 'missing_official_result_provenance'}"
        )
    manifest_path = str(payload.get("official_execution_manifest_path") or "")
    if not manifest_path:
        raise RuntimeError(f"official_result_bundle_missing_execution_manifest:{candidate}")
    if not Path(manifest_path).exists():
        raise RuntimeError(f"official_result_bundle_execution_manifest_missing:{candidate}:{manifest_path}")
    expected_baseline_id = str(baseline_id or "").strip()
    if expected_baseline_id:
        validate_official_bundle_baseline_identity(payload, candidate, baseline_id=expected_baseline_id)


def read_official_result_bundle_if_available(
    *,
    baseline_id: str,
    args: argparse.Namespace,
    source_dir: Path,
    output_json_path: Path,
) -> dict[str, Any] | None:
    """读取 Google Drive 中由本项目 workflow 生成的官方 baseline 结果包。

    这一实现属于项目特定写法。它解决的问题是: 部分现代视频水印 baseline
    需要特定生成模型、训练出的 extractor 或高显存环境, 不能保证在同一个
    Wan2.1 Colab 会话中即时复跑。正式论文比较允许读取项目内 official bundle
    cache, 但必须满足三个条件:

    1. JSON 中存在可审计 score 字段。
    2. `official_result_provenance` 必须是
       `repository_generated_from_third_party_official_code`。
    3. `official_execution_manifest_path` 必须存在, 用于证明该 bundle 来自项目内
       clone / build / run / adapt 链路。
    4. 必须显式记录 official score extraction policy、score semantics 和
       prompt / seed / attack anchor, 避免公平校准时静默混用不同分数口径。
    """
    if os.environ.get("SSTW_DISABLE_OFFICIAL_RESULT_BUNDLE_READ", "").strip().lower() in {"1", "true", "yes"}:
        return None
    for candidate in official_bundle_candidate_paths(baseline_id=baseline_id, args=args):
        if not candidate.exists():
            continue
        payload = read_json(candidate)
        validate_score_payload(payload)
        validate_repository_generated_bundle(payload, candidate, baseline_id=baseline_id)
        validate_complete_official_bundle_baseline_identity(payload, candidate, baseline_id=baseline_id)
        validate_clean_negative_payload(payload)
        validate_official_score_extraction_payload(payload)
        validate_official_formal_comparison_eligibility(payload)
        return {
            **payload,
            **official_score_formal_comparison_summary(payload),
            **official_score_formal_comparison_summary(payload, clean_negative=True),
            "official_adapter_status": "measured_from_official_result_bundle",
            "official_adapter_baseline_id": baseline_id,
            "official_baseline_id": baseline_id,
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
    score, _field_name = extract_raw_detector_score(payload)
    return score


def validate_score_payload(payload: Mapping[str, Any]) -> None:
    """确认输出中存在至少一个可审计分数字段。"""
    extract_score(payload)


def validate_clean_negative_payload(payload: Mapping[str, Any]) -> None:
    """确认 official bundle 包含同方法 clean negative 校准分数。

    paper profile 的公平比较不再允许只给 attacked positive 分数。每个
    external baseline 的 official bundle 必须同时携带该 baseline 自己的
    clean negative detector score, 后续才能在相同 target FPR 下校准阈值。
    """

    score = payload.get(
        "external_baseline_clean_negative_score",
        payload.get("clean_negative_score", payload.get("clean_negative_raw_detector_score")),
    )
    if score in {None, "", "unsupported"}:
        raise RuntimeError("official_result_bundle_missing_clean_negative_score")
    try:
        float(score)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"official_result_bundle_invalid_clean_negative_score:{score}") from exc
    video_path = payload.get(
        "external_baseline_clean_negative_video_path",
        payload.get("clean_negative_video_path"),
    )
    if video_path in {None, ""}:
        raise RuntimeError("official_result_bundle_missing_clean_negative_video_path")


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
    validate_clean_negative_payload(payload)
    validate_official_score_extraction_payload(payload)
    validate_official_formal_comparison_eligibility(payload)
    payload = with_official_adapter_identity(
        payload,
        baseline_id=baseline_id,
        candidate=output_json_path,
    )
    enriched = {
        **payload,
        **official_score_formal_comparison_summary(payload),
        **official_score_formal_comparison_summary(payload, clean_negative=True),
        "official_adapter_status": "measured_by_native_official_command",
        "official_adapter_baseline_id": baseline_id,
        "official_baseline_id": baseline_id,
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
    require_paths_exist([args.source_video, args.attacked_video], label="baseline_input_video")
    output_json_path = Path(args.official_output_json)
    source_dir = Path(args.official_source_dir)

    bundled = read_official_result_bundle_if_available(
        baseline_id=baseline_id,
        args=args,
        source_dir=source_dir,
        output_json_path=output_json_path,
    )
    if bundled is not None:
        write_json(output_json_path, bundled)
        print(json.dumps(bundled, ensure_ascii=False, indent=2, sort_keys=True))
        return

    source_dir = verify_official_source(args.official_source_dir, required_source_files)

    payload = run_native_command_if_configured(
        baseline_id=baseline_id,
        args=args,
        output_json_path=output_json_path,
    )
    if payload is None:
        payload = default_runner(args, source_dir, output_json_path)
        validate_score_payload(payload)
        validate_clean_negative_payload(payload)
        validate_official_score_extraction_payload(payload)
        validate_official_formal_comparison_eligibility(payload)
        payload = {
            **payload,
            **official_score_formal_comparison_summary(payload),
            **official_score_formal_comparison_summary(payload, clean_negative=True),
        }
        payload = with_official_adapter_identity(
            payload,
            baseline_id=baseline_id,
            candidate=output_json_path,
        )
        write_json(output_json_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def raise_missing_official_artifacts(baseline_id: str, details: str) -> None:
    """在官方权重或官方中间产物缺失时给出明确失败原因。"""
    raise RuntimeError(
        f"{baseline_id}_official_required_artifacts_missing:{details}. "
        "请配置对应 SSTW_<BASELINE>_NATIVE_EVAL_COMMAND, 或在 Google Drive 中提供官方权重、"
        "官方 key / message / maintained info 等可复现实验产物。"
    )
