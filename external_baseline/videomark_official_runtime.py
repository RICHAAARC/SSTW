"""VideoMark 官方 embedding / extraction / temporal tamper 流程的项目内运行器。

该模块把第三方 VideoMark 官方仓库纳入 SSTW 的自包含 external baseline
闭环: clone / build / run / adapt / record。它不直接写论文正式
`measured_formal` records, 而是生成 official bundle 和执行 manifest, 后续仍由
统一 `external_baseline_runner` 转写正式记录。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping

from external_baseline.official_eval_adapters.common import REPOSITORY_GENERATED_OFFICIAL_PROVENANCE
from external_baseline.runtime_trace_io import build_comparison_unit_id, comparable_detection_records


BASELINE_ID = "videomark"
DEFAULT_MODEL_NAME = "modelscope"
DEFAULT_NUM_FRAMES = 16
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_NUM_BIT = 512
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_PROMPT_VARIANTS = 1
DEFAULT_DETECTION_THRESHOLD = 0.5
EMBEDDING_VARIANT_LOOP_TARGET = "    for item in tqdm(range(4)):"
EMBEDDING_VARIANT_LOOP_REPLACEMENT = (
    "    variant_count = int(os.environ.get(\"SSTW_VIDEOMARK_PROMPT_VARIANTS\", \"4\"))\n"
    "    for item in tqdm(range(variant_count)):"
)
EMBEDDING_UNDETECTED_TARGET = (
    "    if not detection_result:\n"
    "        decode_message_str = message_placeholder\n"
    "    else:\n"
    "        decode_message = Decode(decoding_key, reversed_prc)\n"
    "        decode_message_str = bits_to_string(decode_message)"
)
EMBEDDING_UNDETECTED_REPLACEMENT = (
    "    if not detection_result:\n"
    "        decode_message = np.full((len(message_bits[0]),), -1)\n"
    "        decode_message_str = message_placeholder\n"
    "    else:\n"
    "        decode_message = Decode(decoding_key, reversed_prc)\n"
    "        decode_message_str = bits_to_string(decode_message)"
)
EMBEDDING_MODEL_PATH_ARG_TARGET = "    parser.add_argument('--model_name', default='i2vgen-xl')"
EMBEDDING_MODEL_PATH_ARG_REPLACEMENT = (
    "    parser.add_argument('--model_name', default='i2vgen-xl')\n"
    "    parser.add_argument('--model_path', default=None)"
)


@dataclass(frozen=True)
class VideoMarkOfficialRuntimeConfig:
    """VideoMark 官方运行器的显式配置。

    该配置属于项目特定写法。VideoMark 官方流程需要先生成带水印视频, 再执行
    temporal tamper 与提取, 因此必须显式记录源码、模型、prompt、输出包路径和
    运行规模, 避免 Notebook 单元格中隐藏参数。
    """

    run_root: str
    bundle_root: str
    source_dir: str
    output_root: str
    resource_root: str
    prompt_suite_path: str
    repo_root: str = "."
    model_name: str = DEFAULT_MODEL_NAME
    model_path: str = ""
    output_path: str = ""
    max_records: int | None = None
    num_frames: int = DEFAULT_NUM_FRAMES
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_bit: int = DEFAULT_NUM_BIT
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    num_inversion_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    prompt_variants: int = DEFAULT_PROMPT_VARIANTS
    device: str = "cuda:0"
    dry_run: bool = False
    timeout_seconds: float = 0.0
    allow_prompt_id_fallback: bool = False
    force_rebuild_runtime_source: bool = True


def _env_bool(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""

    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量。"""

    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量。"""

    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def _read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 对象, 并兼容 UTF-8 BOM。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"json_payload_must_be_object:{path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写出 JSON artifact。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    """写出文本文件。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _safe_token(value: Any) -> str:
    """把记录字段转换为文件名安全 token。"""

    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _is_relative_to(child: Path, parent: Path) -> bool:
    """兼容不同 Python 版本的路径包含关系判断。"""

    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _replace_directory(path: Path, allowed_root: Path) -> None:
    """删除并重建受控目录。

    该函数属于防御式工程写法。运行器需要重建 runtime source copy 和 prompt set,
    但只能删除明确位于 output_root 内的目录, 避免误删用户文件或官方源码。
    """

    resolved_path = path.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_path == resolved_root or not _is_relative_to(resolved_path, resolved_root):
        raise RuntimeError(f"unsafe_replace_directory:{resolved_path}:allowed_root={resolved_root}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _drive_project_root_from_run_root(run_root: Path) -> Path:
    """从统一 run_root 推断 Google Drive 项目根目录。"""

    parts = list(run_root.parts)
    if "runs" in parts:
        return Path(*parts[: parts.index("runs")])
    return run_root.parents[1] if len(run_root.parents) >= 2 else run_root.parent


def _default_prompt_suite_path(run_root: Path) -> Path:
    """推断当前 workflow profile 对应的 prompt suite。"""

    return _drive_project_root_from_run_root(run_root) / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"


def _default_resource_root(run_root: Path) -> Path:
    """推断 external baseline 资源根目录。"""

    return _drive_project_root_from_run_root(run_root) / "resources" / "external_baseline"


def build_default_videomark_official_config_from_env(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    source_dir: str | Path,
    repo_root: str | Path = ".",
    resource_root: str | Path | None = None,
    max_records: int | None = None,
) -> VideoMarkOfficialRuntimeConfig:
    """从 Colab 环境变量构造 VideoMark 官方运行配置。"""

    root = Path(run_root)
    resources = Path(resource_root) if resource_root else _default_resource_root(root)
    prompt_suite_path = os.environ.get("SSTW_VIDEOMARK_PROMPT_SUITE_PATH", "").strip()
    if not prompt_suite_path:
        prompt_suite_path = str(_default_prompt_suite_path(root))
    output_root = os.environ.get("SSTW_VIDEOMARK_OFFICIAL_RUNTIME_OUTPUT_ROOT", "").strip()
    if not output_root:
        output_root = str(Path(bundle_root) / BASELINE_ID / "official_runtime")
    output_path = os.environ.get("SSTW_VIDEOMARK_OFFICIAL_OUTPUT_DIR", "").strip()
    if not output_path:
        output_path = str(Path(bundle_root) / BASELINE_ID / "official_outputs")
    max_records_text = os.environ.get("SSTW_VIDEOMARK_REFERENCE_MAX_RECORDS", "").strip()
    effective_max_records = int(max_records_text) if max_records_text else max_records
    return VideoMarkOfficialRuntimeConfig(
        run_root=str(root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=output_root,
        resource_root=str(resources),
        prompt_suite_path=prompt_suite_path,
        repo_root=str(repo_root),
        model_name=os.environ.get("SSTW_VIDEOMARK_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME,
        model_path=os.environ.get("SSTW_VIDEOMARK_MODEL_PATH", "").strip(),
        output_path=output_path,
        max_records=effective_max_records,
        num_frames=_env_int("SSTW_VIDEOMARK_NUM_FRAMES", DEFAULT_NUM_FRAMES),
        height=_env_int("SSTW_VIDEOMARK_HEIGHT", DEFAULT_HEIGHT),
        width=_env_int("SSTW_VIDEOMARK_WIDTH", DEFAULT_WIDTH),
        num_bit=_env_int("SSTW_VIDEOMARK_NUM_BIT", DEFAULT_NUM_BIT),
        num_inference_steps=_env_int("SSTW_VIDEOMARK_NUM_INFERENCE_STEPS", DEFAULT_NUM_INFERENCE_STEPS),
        num_inversion_steps=_env_int("SSTW_VIDEOMARK_NUM_INVERSION_STEPS", DEFAULT_NUM_INFERENCE_STEPS),
        prompt_variants=_env_int("SSTW_VIDEOMARK_PROMPT_VARIANTS", DEFAULT_PROMPT_VARIANTS),
        device=os.environ.get("SSTW_VIDEOMARK_DEVICE", "cuda:0").strip() or "cuda:0",
        dry_run=_env_bool("SSTW_VIDEOMARK_OFFICIAL_DRY_RUN", False),
        timeout_seconds=_env_float("SSTW_VIDEOMARK_OFFICIAL_TIMEOUT_SECONDS", 0.0),
        allow_prompt_id_fallback=_env_bool("SSTW_VIDEOMARK_ALLOW_PROMPT_ID_FALLBACK", False),
        force_rebuild_runtime_source=_env_bool("SSTW_VIDEOMARK_FORCE_REBUILD_RUNTIME_SOURCE", True),
    )


def _required_source_files(source_dir: Path) -> list[dict[str, Any]]:
    """检查 VideoMark 官方仓库最小入口文件。"""

    required = ("embedding_and_extraction.py", "temporal_tamper.py", "src/prc.py")
    rows: list[dict[str, Any]] = []
    for relative_path in required:
        path = source_dir / relative_path
        rows.append({
            "relative_path": relative_path,
            "path": str(path),
            "exists": path.exists(),
        })
    return rows


def _ensure_source_ready(source_dir: Path) -> dict[str, Any]:
    """验证官方源码目录。"""

    rows = _required_source_files(source_dir)
    missing = [row["relative_path"] for row in rows if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"videomark_official_source_required_files_missing:{missing}:source_dir={source_dir}")
    return {
        "official_source_dir": str(source_dir),
        "required_source_files": rows,
        "source_status": "ready",
    }


def _copy_official_source_to_runtime(source_dir: Path, runtime_source_dir: Path, output_root: Path, *, force: bool) -> None:
    """复制官方源码到 runtime 工作副本。"""

    if runtime_source_dir.exists() and force:
        _replace_directory(runtime_source_dir, output_root)
        runtime_source_dir.rmdir()
    if runtime_source_dir.exists():
        return

    def _ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "result",
            "results",
            "eval_quality",
            "logs",
        }
        return {name for name in names if name in ignored or name.endswith(".pyc")}

    runtime_source_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, runtime_source_dir, ignore=_ignore)


def _patch_videomark_runtime_source(runtime_source_dir: Path) -> dict[str, Any]:
    """修补 runtime 副本中 VideoMark 官方脚本的 Colab 兼容性。

    该补丁只作用于 runtime 副本, 不修改 checked-in 第三方源码。主要修复两点:
    1. 暴露 `SSTW_VIDEOMARK_PROMPT_VARIANTS`, 使 validation_scale 可控制小样本运行量。
    2. 当官方提取未检测到 watermark 时, 避免 `decode_message` 未定义导致流程中断。
    """

    embedding_path = runtime_source_dir / "embedding_and_extraction.py"
    text = embedding_path.read_text(encoding="utf-8")
    patch_results: list[dict[str, Any]] = []
    if EMBEDDING_VARIANT_LOOP_REPLACEMENT in text:
        patch_results.append({"patch_name": "prompt_variant_count_env_guard", "patch_status": "already_patched"})
    elif EMBEDDING_VARIANT_LOOP_TARGET in text:
        text = text.replace(EMBEDDING_VARIANT_LOOP_TARGET, EMBEDDING_VARIANT_LOOP_REPLACEMENT, 1)
        patch_results.append({"patch_name": "prompt_variant_count_env_guard", "patch_status": "patched_runtime_copy"})
    else:
        patch_results.append({"patch_name": "prompt_variant_count_env_guard", "patch_status": "pattern_missing_no_change"})

    if EMBEDDING_UNDETECTED_REPLACEMENT in text:
        patch_results.append({"patch_name": "undetected_decode_message_guard", "patch_status": "already_patched"})
    elif EMBEDDING_UNDETECTED_TARGET in text:
        text = text.replace(EMBEDDING_UNDETECTED_TARGET, EMBEDDING_UNDETECTED_REPLACEMENT, 1)
        patch_results.append({"patch_name": "undetected_decode_message_guard", "patch_status": "patched_runtime_copy"})
    else:
        patch_results.append({"patch_name": "undetected_decode_message_guard", "patch_status": "pattern_missing_no_change"})

    if "parser.add_argument('--model_path'" in text or 'parser.add_argument("--model_path"' in text:
        patch_results.append({"patch_name": "embedding_model_path_cli_arg_guard", "patch_status": "already_patched"})
    elif EMBEDDING_MODEL_PATH_ARG_TARGET in text:
        text = text.replace(EMBEDDING_MODEL_PATH_ARG_TARGET, EMBEDDING_MODEL_PATH_ARG_REPLACEMENT, 1)
        patch_results.append({"patch_name": "embedding_model_path_cli_arg_guard", "patch_status": "patched_runtime_copy"})
    else:
        patch_results.append({"patch_name": "embedding_model_path_cli_arg_guard", "patch_status": "pattern_missing_no_change"})

    embedding_path.write_text(text, encoding="utf-8")
    status = "patched_runtime_copy" if any(
        row["patch_status"] == "patched_runtime_copy" for row in patch_results
    ) else "already_patched"
    return {
        "patch_name": "videomark_colab_runtime_compatibility",
        "patch_status": status,
        "patch_results": patch_results,
        "patched_file": str(embedding_path),
        "source_mutation_policy": "runtime_copy_only_checked_in_official_source_not_modified",
    }


def _prompt_rows_from_suite(prompt_suite_path: Path) -> dict[str, str]:
    """读取 SSTW prompt suite 中的 prompt_id 到 prompt_text 映射。"""

    suite = _read_json(prompt_suite_path)
    prompts = suite.get("prompts", [])
    if not isinstance(prompts, list):
        raise TypeError(f"prompt_suite_prompts_must_be_list:{prompt_suite_path}")
    rows: dict[str, str] = {}
    for item in prompts:
        if not isinstance(item, Mapping):
            continue
        prompt_id = str(item.get("prompt_id") or "").strip()
        prompt_text = str(item.get("prompt_text") or "").strip()
        if prompt_id and prompt_text:
            rows[prompt_id] = prompt_text
    return rows


def _selected_runtime_records(run_root: Path, max_records: int | None) -> list[dict[str, Any]]:
    """读取本次要锚定的 runtime detection records。"""

    records = comparable_detection_records(run_root)
    if max_records is not None:
        return records[: int(max_records)]
    return records


def _runtime_prompt_rows(
    records: list[Mapping[str, Any]],
    *,
    prompt_suite_path: Path,
    allow_prompt_id_fallback: bool,
) -> list[dict[str, Any]]:
    """按 runtime records 的首次出现顺序构造 VideoMark prompt set。"""

    prompt_text_by_id = _prompt_rows_from_suite(prompt_suite_path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []
    for record in records:
        prompt_id = str(record.get("prompt_id") or "").strip()
        if not prompt_id or prompt_id in seen:
            continue
        prompt_text = prompt_text_by_id.get(prompt_id)
        if not prompt_text:
            if allow_prompt_id_fallback:
                prompt_text = prompt_id
            else:
                missing.append(prompt_id)
                continue
        seen.add(prompt_id)
        rows.append({
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
        })
    if missing:
        raise KeyError(f"videomark_prompt_text_missing_for_prompt_ids:{missing}:prompt_suite={prompt_suite_path}")
    if not rows:
        raise RuntimeError("videomark_no_runtime_prompts_selected")
    return rows


def _write_runtime_prompt_set(
    runtime_source_dir: Path,
    output_root: Path,
    *,
    prompt_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """把 SSTW prompt suite 转成 VideoMark 官方 data/test_prompts.txt。"""

    data_dir = runtime_source_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = data_dir / "test_prompts.txt"
    prompt_texts = [str(row["prompt_text"]).strip() for row in prompt_rows]
    _write_text(prompt_file, "\n".join(prompt_texts) + "\n")
    meta_file = data_dir / "sstw_runtime_prompt_meta.json"
    meta_payload = [
        {
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "sstw_prompt_anchor": "same_prompt_id_from_runtime_detection_records",
        }
        for row in prompt_rows
    ]
    _write_json(meta_file, {"prompts": meta_payload})
    manifest = {
        "manifest_kind": "videomark_runtime_prompt_set_manifest",
        "prompt_file": str(prompt_file),
        "meta_file": str(meta_file),
        "prompt_count": len(prompt_rows),
        "prompt_ids": [row["prompt_id"] for row in prompt_rows],
        "prompt_set_policy": "same_prompt_anchor_for_external_baseline_official_flow",
    }
    _write_json(output_root / "videomark_runtime_prompt_set_manifest.json", manifest)
    return manifest


def _videomark_embedding_command(config: VideoMarkOfficialRuntimeConfig) -> list[str]:
    """构造 VideoMark 官方 embedding_and_extraction.py 命令。"""

    command = [
        sys.executable,
        "embedding_and_extraction.py",
        f"--device={config.device}",
        f"--model_name={config.model_name}",
        f"--num_frames={int(config.num_frames)}",
        f"--height={int(config.height)}",
        f"--width={int(config.width)}",
        f"--num_bit={int(config.num_bit)}",
        f"--num_inference_steps={int(config.num_inference_steps)}",
        f"--num_inversion_steps={int(config.num_inversion_steps)}",
        f"--output_dir={config.output_path}",
        "--data_dir=data",
        "--keys_path=./keys",
    ]
    if config.model_path:
        command.append(f"--model_path={config.model_path}")
    return command


def _videomark_temporal_tamper_command(config: VideoMarkOfficialRuntimeConfig) -> list[str]:
    """构造 VideoMark 官方 temporal_tamper.py 命令。"""

    command = [
        sys.executable,
        "temporal_tamper.py",
        f"--device={config.device}",
        f"--model_name={config.model_name}",
        f"--num_bit={int(config.num_bit)}",
        f"--num_inversion_steps={int(config.num_inversion_steps)}",
        f"--video_frames_dir={config.output_path}",
        "--keys_path=./keys",
    ]
    if config.model_path:
        command.append(f"--model_path={config.model_path}")
    return command


def _run_videomark_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    prompt_variants: int,
) -> dict[str, Any]:
    """运行单条 VideoMark 官方命令并写出 stdout / stderr。"""

    env = dict(os.environ)
    env["SSTW_VIDEOMARK_PROMPT_VARIANTS"] = str(int(prompt_variants))
    timeout_expired = False
    execution_error = ""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        # VideoMark 官方模型下载和生成耗时较长。若用户设置 timeout, 这里必须写出
        # governed failure 日志, 而不是让 Notebook 直接中断导致缺少决策 artifact。
        timeout_expired = True
        stdout = _text_from_subprocess_payload(exc.stdout)
        stderr = _text_from_subprocess_payload(exc.stderr) + f"\ncommand_timeout_seconds={timeout_seconds}"
        return_code = -9
    except OSError as exc:
        execution_error = str(exc)
        stdout = ""
        stderr = f"videomark_official_command_launch_error:{execution_error}"
        return_code = -1
    stdout_path = log_path.with_name(log_path.name + "_stdout.txt")
    stderr_path = log_path.with_name(log_path.name + "_stderr.txt")
    _write_text(stdout_path, stdout)
    _write_text(stderr_path, stderr)
    return {
        "command": command,
        "cwd": str(cwd),
        "return_code": return_code,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "timeout_expired": timeout_expired,
        "execution_error": execution_error,
    }


def _text_from_subprocess_payload(value: Any) -> str:
    """把 subprocess 异常中的 stdout / stderr 规范化为文本。"""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _videomark_result_dir(output_path: Path, model_name: str, num_bit: int) -> Path:
    """返回 VideoMark 官方结果目录。"""

    return output_path / "videomark" / model_name / f"{int(num_bit)}bit"


def _collect_metric_values(payload: Any, key_name: str) -> list[float]:
    """递归收集官方 JSON 中的指定数值字段。"""

    values: list[float] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == key_name:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            else:
                values.extend(_collect_metric_values(value, key_name))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_metric_values(item, key_name))
    return values


def _collect_temporal_attack_names(payload: Any) -> list[str]:
    """收集 VideoMark temporal_results 中出现的官方时序攻击名称。"""

    names: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, Mapping) and ("decode_acc" in value or "frames_acc" in value):
                names.add(str(key))
            else:
                names.update(_collect_temporal_attack_names(value))
    elif isinstance(payload, list):
        for item in payload:
            names.update(_collect_temporal_attack_names(item))
    return sorted(names)


def _score_from_videomark_temporal_results(temporal_results_path: Path) -> dict[str, Any]:
    """从 VideoMark 官方 temporal_results.json 计算聚合 score。"""

    payload = _read_json(temporal_results_path)
    decode_values = _collect_metric_values(payload, "decode_acc")
    frame_values = _collect_metric_values(payload, "frames_acc")
    if not decode_values:
        raise RuntimeError(f"videomark_temporal_decode_acc_missing:{temporal_results_path}")
    decode_mean = sum(decode_values) / len(decode_values)
    frame_mean = sum(frame_values) / len(frame_values) if frame_values else None
    result: dict[str, Any] = {
        "bit_accuracy": round(float(decode_mean), 6),
        "external_baseline_score": round(float(decode_mean), 6),
        "official_decode_acc_count": len(decode_values),
        "official_decode_acc_mean": round(float(decode_mean), 6),
        "official_temporal_attack_names": _collect_temporal_attack_names(payload),
    }
    if frame_mean is not None:
        result["official_frames_acc_count"] = len(frame_values)
        result["official_frames_acc_mean"] = round(float(frame_mean), 6)
    return result


def _bundle_record_path(bundle_root: Path, record: Mapping[str, Any]) -> Path:
    """构造 VideoMark official bundle 的单条记录路径。"""

    prompt = _safe_token(record.get("prompt_id"))
    seed = _safe_token(record.get("seed_id"))
    attack = _safe_token(record.get("attack_name"))
    return bundle_root / BASELINE_ID / "records" / f"{prompt}__{seed}__{attack}.json"


def write_videomark_official_bundle_records(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    manifest_path: str | Path,
    temporal_results_json_path: str | Path,
    video_results_json_path: str | Path,
    model_name: str,
    max_records: int | None = None,
) -> dict[str, Any]:
    """把 VideoMark 官方 temporal_results 输出转写为 official bundle records。

    这是适配层, 不是正式论文计分层。它只把官方 temporal tamper decode accuracy
    以同一 prompt / seed / attack 锚点写入项目内 official bundle, 后续统一
    runner 才会生成 `metric_status: measured_formal` 的正式记录。
    """

    root = Path(run_root)
    bundle = Path(bundle_root)
    temporal_path = Path(temporal_results_json_path)
    video_path = Path(video_results_json_path)
    score_payload = _score_from_videomark_temporal_results(temporal_path)
    records = _selected_runtime_records(root, max_records)
    generated = 0
    failures: list[dict[str, Any]] = []
    threshold = float(os.environ.get("SSTW_VIDEOMARK_DETECTION_THRESHOLD", str(DEFAULT_DETECTION_THRESHOLD)))
    for record in records:
        output_json_path = _bundle_record_path(bundle, record)
        try:
            payload = {
                **score_payload,
                "detected": float(score_payload["external_baseline_score"]) >= threshold,
                "threshold": threshold,
                "official_result_provenance": REPOSITORY_GENERATED_OFFICIAL_PROVENANCE,
                "official_baseline_id": BASELINE_ID,
                "external_baseline_generation_model_id": model_name,
                "external_baseline_official_execution_mode": "videomark_embedding_extraction_temporal_tamper",
                "official_score_assignment_policy": "aggregate_mean_over_videomark_official_temporal_results_json",
                "attack_protocol_status": "videomark_official_temporal_tamper_score_reused_for_runtime_attack_anchor",
                "official_temporal_results_json_path": str(temporal_path),
                "official_video_results_json_path": str(video_path),
                "official_execution_manifest_path": str(manifest_path),
                "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                "runtime_comparison_unit_id": build_comparison_unit_id(BASELINE_ID, record),
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "trajectory_trace_id": record.get("trajectory_trace_id"),
                "source_video_path": record.get("source_video_path"),
                "attacked_video_path": record.get("attacked_video_path"),
                "claim_support_status": "official_reference_bundle_written_not_claim_by_itself",
            }
            _write_json(output_json_path, payload)
            generated += 1
        except Exception as exc:  # pragma: no cover - 单条文件系统异常难以稳定复现
            failures.append({
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "failure_reason": str(exc),
            })
    return {
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": generated,
        "failed_bundle_record_count": len(failures),
        "failures": failures[:20],
    }


def run_videomark_official_runtime(config: VideoMarkOfficialRuntimeConfig) -> dict[str, Any]:
    """执行 VideoMark 官方 embedding / extraction / temporal tamper 并生成 official bundle。"""

    run_root = Path(config.run_root)
    bundle_root = Path(config.bundle_root)
    source_dir = Path(config.source_dir)
    output_root = Path(config.output_root)
    output_path = Path(config.output_path)
    prompt_suite_path = Path(config.prompt_suite_path)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    source_audit = _ensure_source_ready(source_dir)
    records = _selected_runtime_records(run_root, config.max_records)
    if not records:
        raise RuntimeError(f"videomark_runtime_detection_records_missing:{run_root / 'records/runtime_detection_records.jsonl'}")
    prompt_rows = _runtime_prompt_rows(
        records,
        prompt_suite_path=prompt_suite_path,
        allow_prompt_id_fallback=config.allow_prompt_id_fallback,
    )
    runtime_source_dir = output_root / "source_runtime"
    _copy_official_source_to_runtime(
        source_dir,
        runtime_source_dir,
        output_root,
        force=config.force_rebuild_runtime_source,
    )
    patch_manifest = _patch_videomark_runtime_source(runtime_source_dir)
    prompt_manifest = _write_runtime_prompt_set(
        runtime_source_dir,
        output_root,
        prompt_rows=prompt_rows,
    )
    manifest_path = bundle_root / BASELINE_ID / "official_reference_execution_manifest.json"
    result_dir = _videomark_result_dir(output_path, config.model_name, config.num_bit)
    video_results_path = result_dir / "video_results.json"
    temporal_results_path = result_dir / "temporal_results.json"
    embedding_command = _videomark_embedding_command(config)
    temporal_tamper_command = _videomark_temporal_tamper_command(config)
    command_results: list[dict[str, Any]] = []
    execution_status = "dry_run_planned" if config.dry_run else "executed"
    execution_failure_reason = ""

    if not config.dry_run:
        embedding_result = _run_videomark_command(
            embedding_command,
            cwd=runtime_source_dir,
            log_path=output_root / "logs" / "videomark_embedding_and_extraction",
            timeout_seconds=float(config.timeout_seconds),
            prompt_variants=int(config.prompt_variants),
        )
        command_results.append(embedding_result)
        if int(embedding_result["return_code"]) != 0:
            execution_failure_reason = (
                f"videomark_official_embedding_failed:{embedding_result['return_code']}:{embedding_result['stderr_tail']}"
            )
        if not execution_failure_reason:
            temporal_result = _run_videomark_command(
                temporal_tamper_command,
                cwd=runtime_source_dir,
                log_path=output_root / "logs" / "videomark_temporal_tamper",
                timeout_seconds=float(config.timeout_seconds),
                prompt_variants=int(config.prompt_variants),
            )
            command_results.append(temporal_result)
            if int(temporal_result["return_code"]) != 0:
                execution_failure_reason = (
                    f"videomark_official_temporal_tamper_failed:{temporal_result['return_code']}:{temporal_result['stderr_tail']}"
                )

    bundle_result = {
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": 0,
        "failed_bundle_record_count": 0,
        "failures": [],
    }
    if not config.dry_run and not execution_failure_reason:
        if temporal_results_path.is_file():
            os.environ["SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON"] = str(temporal_results_path)
            bundle_result = write_videomark_official_bundle_records(
                run_root=run_root,
                bundle_root=bundle_root,
                manifest_path=manifest_path,
                temporal_results_json_path=temporal_results_path,
                video_results_json_path=video_results_path,
                model_name=config.model_name,
                max_records=config.max_records,
            )
        else:
            execution_failure_reason = f"videomark_temporal_results_json_missing:{temporal_results_path}"
    if execution_failure_reason:
        execution_status = "failed"
        bundle_result = {
            "input_runtime_detection_record_count": len(records),
            "generated_bundle_record_count": 0,
            "failed_bundle_record_count": len(records),
            "failures": [
                {
                    "baseline_id": BASELINE_ID,
                    "prompt_id": record.get("prompt_id"),
                    "seed_id": record.get("seed_id"),
                    "attack_name": record.get("attack_name"),
                    "failure_reason": execution_failure_reason,
                }
                for record in records[:20]
            ],
        }

    manifest = {
        "manifest_kind": "modern_external_baseline_formal_reference_execution_manifest",
        "baseline_id": BASELINE_ID,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "official_source_dir": str(source_dir),
        "runtime_source_dir": str(runtime_source_dir),
        "official_runtime_output_root": str(output_root),
        "official_output_path": str(output_path),
        "official_repository_url": "https://github.com/KYRIE-LI11/VideoMark",
        "official_execution_mode": "videomark_embedding_extraction_temporal_tamper",
        "execution_status": execution_status,
        "execution_failure_reason": execution_failure_reason,
        "dry_run": bool(config.dry_run),
        "config": asdict(config),
        "source_audit": source_audit,
        "patch_manifest": patch_manifest,
        "prompt_manifest": prompt_manifest,
        "embedding_command": embedding_command,
        "temporal_tamper_command": temporal_tamper_command,
        "command_results": command_results,
        "video_results_json_path": str(video_results_path),
        "temporal_results_json_path": str(temporal_results_path),
        "temporal_results_json_exists": temporal_results_path.is_file(),
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": int(bundle_result["generated_bundle_record_count"]),
        "failed_bundle_record_count": int(bundle_result["failed_bundle_record_count"]),
        "failures": bundle_result.get("failures", []),
        "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
    }
    _write_json(manifest_path, manifest)
    _write_json(output_root / "videomark_official_execution_manifest.json", manifest)
    if temporal_results_path.is_file():
        os.environ["SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON"] = str(temporal_results_path)
    os.environ["SSTW_VIDEOMARK_OFFICIAL_OUTPUT_DIR"] = str(output_path)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行入口参数。"""

    parser = argparse.ArgumentParser(description="运行 VideoMark 官方 embedding / extraction / temporal tamper 并生成 official bundle")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--resource-root", default="")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """命令行入口。"""

    args = _build_arg_parser().parse_args()
    config = build_default_videomark_official_config_from_env(
        run_root=args.run_root,
        bundle_root=args.bundle_root,
        source_dir=args.source_dir,
        repo_root=args.repo_root,
        resource_root=args.resource_root or None,
        max_records=args.max_records,
    )
    if args.dry_run:
        config = VideoMarkOfficialRuntimeConfig(**{**asdict(config), "dry_run": True})
    result = run_videomark_official_runtime(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI 入口由 Notebook 调用
    main()
