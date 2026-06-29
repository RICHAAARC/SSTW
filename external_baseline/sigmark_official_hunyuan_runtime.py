"""SIGMark 官方 Hunyuan gen->extract 流程的项目内运行器。

该模块的职责是把第三方 SIGMark 官方仓库纳入 SSTW 的自包含 external baseline
闭环: clone / build / run / adapt / record。它不直接产出论文 `measured_formal`
记录, 而是生成 official bundle 和执行 manifest, 后续仍由统一
`external_baseline_runner` 转写为正式记录。
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


BASELINE_ID = "sigmark"
DEFAULT_MODEL_NAME = "HunyuanVideo-community"
DEFAULT_PROMPT_SET = "VBench2_aug"
DEFAULT_RUNTIME_DIMENSION = "sstw_runtime_prompt"
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_PRECISION = "bf16"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_NUM_FRAMES = 65
DEFAULT_FPS = 8
DEFAULT_CH_FACTOR = 2
DEFAULT_HW_FACTOR = 8
DEFAULT_FR_FACTOR = 1
DEFAULT_VAE_SCALE_FACTOR_SPATIAL = 8
DEFAULT_VAE_SCALE_FACTOR_TEMPORAL = 4
DEFAULT_LATENT_CHANNELS = 16
PRECISION_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}
OFFICIAL_PRECISION_CHOICES = {"fp16", "bf16", "fp32"}
PATCH_TARGET = (
    '        image_prompt = load_image(os.path.join(args.image_prompt_dir, dimension, prompt[:180] + "-0.png")) \\\n'
    "            if args.image_prompt_dir is not None else None"
)
PATCH_REPLACEMENT = (
    '        image_prompt = load_image(os.path.join(args.image_prompt_dir, dimension, prompt[:180] + "-0.png")) \\\n'
    '            if "I2V" in args.model_name and args.image_prompt_dir is not None else None'
)
EXTRACT_VALID_INDEX_TARGET = "        else:\n            valid_index = None"
EXTRACT_VALID_INDEX_REPLACEMENT = "        else:\n            valid_index = [None] * len(sample_names)"
HF_MODEL_REPOS = {
    "HunyuanVideo-community": "hunyuanvideo-community/HunyuanVideo",
    "HunyuanVideo-I2V-community": "hunyuanvideo-community/HunyuanVideo-I2V",
}


@dataclass(frozen=True)
class SigmarkOfficialHunyuanRuntimeConfig:
    """SIGMark 官方运行器的显式配置。

    该配置属于项目特定写法。SIGMark 官方流程需要生成视频再盲提取 watermark,
    因此必须把源码、模型、prompt、输出包路径和运行规模全部显式化, 避免 Notebook
    单元格中隐藏参数。
    """

    run_root: str
    bundle_root: str
    source_dir: str
    output_root: str
    resource_root: str
    prompt_suite_path: str
    repo_root: str = "."
    model_name: str = DEFAULT_MODEL_NAME
    model_base_path: str = ""
    prompt_set: str = DEFAULT_PROMPT_SET
    runtime_dimension: str = DEFAULT_RUNTIME_DIMENSION
    output_path: str = ""
    max_records: int | None = None
    num_videos_per_prompt: int = 1
    num_videos_per_prompt_diversity: int = 1
    batch_size: int = 1
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    num_frames: int = DEFAULT_NUM_FRAMES
    fps: int = DEFAULT_FPS
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    seed: int = 42
    ch_factor: int = DEFAULT_CH_FACTOR
    hw_factor: int = DEFAULT_HW_FACTOR
    fr_factor: int = DEFAULT_FR_FACTOR
    precision: str = DEFAULT_PRECISION
    quant_text_encoder: int = 1
    nproc_per_node: int = 1
    small_scale_test: int = -1
    auto_download_hf_model: bool = False
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


def normalize_sigmark_precision(value: str) -> str:
    """把常见 dtype 名称转换为 SIGMark 官方 CLI 可接受的 precision token。

    SIGMark 官方 `main.py` 只接受 `fp16`、`bf16` 和 `fp32`。Colab 与 PyTorch
    文档中常见的 `bfloat16` 属于 dtype 名称, 不能直接传给官方 CLI。因此这里在
    项目运行器层做显式归一化, 避免 Notebook 冷启动运行到官方 argparse 时才失败。
    """

    normalized = PRECISION_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())
    if normalized not in OFFICIAL_PRECISION_CHOICES:
        raise ValueError(f"sigmark_precision_invalid:{value}:choices={sorted(OFFICIAL_PRECISION_CHOICES)}")
    return normalized


def validate_sigmark_watermark_geometry(
    *,
    width: int,
    height: int,
    num_frames: int,
    ch_factor: int,
    hw_factor: int,
    fr_factor: int,
    vae_scale_factor_spatial: int = DEFAULT_VAE_SCALE_FACTOR_SPATIAL,
    vae_scale_factor_temporal: int = DEFAULT_VAE_SCALE_FACTOR_TEMPORAL,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
) -> dict[str, Any]:
    """在加载 Hunyuan 模型前校验 SIGMark 官方水印几何约束。

    该函数属于项目特定防御式写法。SIGMark 官方代码会在构造 watermark 时要求
    latent height / width 能被 `hw_factor` 整除。若使用 720x1280 且 `hw_factor=8`,
    latent width 为 90, 会在官方模型已经加载后才失败。此处提前校验, 并把默认值
    固定为官方 `main.py` 的 512x512 / 65 frames / ch=2 / hw=8 / fr=1。
    """

    if width <= 0 or height <= 0 or num_frames <= 0:
        raise ValueError(f"sigmark_watermark_geometry_invalid:non_positive_size:{width}x{height}x{num_frames}")
    if ch_factor <= 0 or hw_factor <= 0 or fr_factor <= 0:
        raise ValueError(
            "sigmark_watermark_geometry_invalid:non_positive_factor:"
            f"ch_factor={ch_factor}:hw_factor={hw_factor}:fr_factor={fr_factor}"
        )
    violations: list[str] = []
    if height % vae_scale_factor_spatial != 0 or width % vae_scale_factor_spatial != 0:
        violations.append("video_hw_not_divisible_by_vae_scale_factor_spatial")
    if (num_frames - 1) % vae_scale_factor_temporal != 0:
        violations.append("video_frames_not_divisible_by_vae_scale_factor_temporal")
    latent_h = height // vae_scale_factor_spatial
    latent_w = width // vae_scale_factor_spatial
    latent_f = (num_frames - 1) // vae_scale_factor_temporal
    if latent_h % hw_factor != 0 or latent_w % hw_factor != 0:
        violations.append("latent_hw_not_divisible_by_hw_factor")
    if latent_f % fr_factor != 0:
        violations.append("latent_f_not_divisible_by_fr_factor")
    if latent_channels % ch_factor != 0:
        violations.append("latent_channels_not_divisible_by_ch_factor")
    manifest = {
        "geometry_status": "ready" if not violations else "invalid",
        "width": int(width),
        "height": int(height),
        "num_frames": int(num_frames),
        "ch_factor": int(ch_factor),
        "hw_factor": int(hw_factor),
        "fr_factor": int(fr_factor),
        "vae_scale_factor_spatial": int(vae_scale_factor_spatial),
        "vae_scale_factor_temporal": int(vae_scale_factor_temporal),
        "latent_channels": int(latent_channels),
        "latent_h": int(latent_h),
        "latent_w": int(latent_w),
        "latent_f": int(latent_f),
        "required_width_multiple": int(vae_scale_factor_spatial * hw_factor),
        "required_height_multiple": int(vae_scale_factor_spatial * hw_factor),
        "required_frame_rule": f"num_frames = {vae_scale_factor_temporal} * n + 1",
        "violations": violations,
    }
    if violations:
        raise ValueError(f"sigmark_watermark_geometry_invalid:{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}")
    return manifest


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
    """推断当前 validation_scale / pilot_paper profile 对应的 prompt suite。"""

    return _drive_project_root_from_run_root(run_root) / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"


def _default_resource_root(run_root: Path) -> Path:
    """推断 external baseline 资源根目录。"""

    return _drive_project_root_from_run_root(run_root) / "resources" / "external_baseline"


def build_default_sigmark_official_hunyuan_config_from_env(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    source_dir: str | Path,
    repo_root: str | Path = ".",
    resource_root: str | Path | None = None,
    max_records: int | None = None,
) -> SigmarkOfficialHunyuanRuntimeConfig:
    """从 Colab 环境变量构造 SIGMark 官方 Hunyuan 运行配置。"""

    root = Path(run_root)
    resources = Path(resource_root) if resource_root else _default_resource_root(root)
    model_name = os.environ.get("SSTW_SIGMARK_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    model_base_path = os.environ.get("SSTW_SIGMARK_MODEL_BASE_PATH", "").strip()
    if not model_base_path:
        model_base_path = str(resources / BASELINE_ID / "models")
    prompt_suite_path = os.environ.get("SSTW_SIGMARK_PROMPT_SUITE_PATH", "").strip()
    if not prompt_suite_path:
        prompt_suite_path = str(_default_prompt_suite_path(root))
    output_root = os.environ.get("SSTW_SIGMARK_OFFICIAL_RUNTIME_OUTPUT_ROOT", "").strip()
    if not output_root:
        output_root = str(Path(bundle_root) / BASELINE_ID / "official_hunyuan_runtime")
    output_path = os.environ.get("SSTW_SIGMARK_OFFICIAL_OUTPUT_DIR", "").strip()
    if not output_path:
        output_path = str(Path(bundle_root) / BASELINE_ID / "official_hunyuan_outputs")
    max_records_text = os.environ.get("SSTW_SIGMARK_REFERENCE_MAX_RECORDS", "").strip()
    effective_max_records = int(max_records_text) if max_records_text else max_records
    timeout = _env_float("SSTW_SIGMARK_OFFICIAL_TIMEOUT_SECONDS", 0.0)
    return SigmarkOfficialHunyuanRuntimeConfig(
        run_root=str(root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=output_root,
        resource_root=str(resources),
        prompt_suite_path=prompt_suite_path,
        repo_root=str(repo_root),
        model_name=model_name,
        model_base_path=model_base_path,
        prompt_set=os.environ.get("SSTW_SIGMARK_PROMPT_SET", DEFAULT_PROMPT_SET).strip() or DEFAULT_PROMPT_SET,
        runtime_dimension=os.environ.get("SSTW_SIGMARK_RUNTIME_DIMENSION", DEFAULT_RUNTIME_DIMENSION).strip()
        or DEFAULT_RUNTIME_DIMENSION,
        output_path=output_path,
        max_records=effective_max_records,
        num_videos_per_prompt=_env_int("SSTW_SIGMARK_NUM_VIDEOS_PER_PROMPT", 1),
        num_videos_per_prompt_diversity=_env_int("SSTW_SIGMARK_NUM_VIDEOS_PER_PROMPT_DIVERSITY", 1),
        batch_size=_env_int("SSTW_SIGMARK_BATCH_SIZE", 1),
        width=_env_int("SSTW_SIGMARK_WIDTH", DEFAULT_WIDTH),
        height=_env_int("SSTW_SIGMARK_HEIGHT", DEFAULT_HEIGHT),
        num_frames=_env_int("SSTW_SIGMARK_NUM_FRAMES", DEFAULT_NUM_FRAMES),
        fps=_env_int("SSTW_SIGMARK_FPS", DEFAULT_FPS),
        guidance_scale=_env_float("SSTW_SIGMARK_GUIDANCE_SCALE", 6.0),
        num_inference_steps=_env_int("SSTW_SIGMARK_NUM_INFERENCE_STEPS", 50),
        seed=_env_int("SSTW_SIGMARK_SEED", 42),
        ch_factor=_env_int("SSTW_SIGMARK_CH_FACTOR", DEFAULT_CH_FACTOR),
        hw_factor=_env_int("SSTW_SIGMARK_HW_FACTOR", DEFAULT_HW_FACTOR),
        fr_factor=_env_int("SSTW_SIGMARK_FR_FACTOR", DEFAULT_FR_FACTOR),
        precision=normalize_sigmark_precision(os.environ.get("SSTW_SIGMARK_PRECISION", DEFAULT_PRECISION)),
        quant_text_encoder=_env_int("SSTW_SIGMARK_QUANT_TEXT_ENCODER", 1),
        nproc_per_node=_env_int("SSTW_SIGMARK_NPROC_PER_NODE", 1),
        small_scale_test=_env_int("SSTW_SIGMARK_SMALL_SCALE_TEST", -1),
        auto_download_hf_model=_env_bool("SSTW_SIGMARK_AUTO_DOWNLOAD_HF_MODEL", False),
        dry_run=_env_bool("SSTW_SIGMARK_OFFICIAL_HUNYUAN_DRY_RUN", False),
        timeout_seconds=timeout,
        allow_prompt_id_fallback=_env_bool("SSTW_SIGMARK_ALLOW_PROMPT_ID_FALLBACK", False),
        force_rebuild_runtime_source=_env_bool("SSTW_SIGMARK_FORCE_REBUILD_RUNTIME_SOURCE", True),
    )


def _required_source_files(source_dir: Path) -> list[dict[str, Any]]:
    """检查 SIGMark 官方仓库最小入口文件。"""

    required = ("main.py", "watermarks/sigmark.py", "apply_disturbances.py")
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
        raise FileNotFoundError(f"sigmark_official_source_required_files_missing:{missing}:source_dir={source_dir}")
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
        ignored = {".git", "__pycache__", ".pytest_cache", "outputs", "output", "logs"}
        return {name for name in names if name in ignored or name.endswith(".pyc")}

    runtime_source_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, runtime_source_dir, ignore=_ignore)


def _patch_sigmark_main_for_t2v(runtime_source_dir: Path) -> dict[str, Any]:
    """修补 runtime 副本中官方 main.py 的 Colab T2V 兼容性。

    官方 main.py 在 T2V 路径中也会先读取 image prompt。SSTW validation_scale
    需要用同一批文本 prompt 锚定外部 baseline, 因而只在 runtime 副本中把 image
    prompt loading 限定到 I2V 模型。同时, 官方 extract 在没有 disturbance_info
    时会把 valid_index 设为 None, 但后续仍将其传入 zip。该路径在 Colab T2V
    官方 gen->extract 中会触发 TypeError, 因此运行器只在 runtime 副本中把它
    改成与 sample_names 等长的占位列表。该补丁不修改 checked-in 第三方源码。
    """

    main_path = runtime_source_dir / "main.py"
    text = main_path.read_text(encoding="utf-8")
    patch_results: list[dict[str, Any]] = []
    if PATCH_REPLACEMENT in text:
        patch_results.append({
            "patch_name": "t2v_image_prompt_load_guard",
            "patch_status": "already_patched",
        })
    elif PATCH_TARGET in text:
        text = text.replace(PATCH_TARGET, PATCH_REPLACEMENT, 1)
        patch_results.append({
            "patch_name": "t2v_image_prompt_load_guard",
            "patch_status": "patched_runtime_copy",
        })
    else:
        raise RuntimeError("sigmark_main_image_prompt_patch_pattern_missing")

    if EXTRACT_VALID_INDEX_REPLACEMENT in text:
        patch_results.append({
            "patch_name": "extract_valid_index_none_guard",
            "patch_status": "already_patched",
        })
    elif EXTRACT_VALID_INDEX_TARGET in text:
        text = text.replace(EXTRACT_VALID_INDEX_TARGET, EXTRACT_VALID_INDEX_REPLACEMENT, 1)
        patch_results.append({
            "patch_name": "extract_valid_index_none_guard",
            "patch_status": "patched_runtime_copy",
        })
    else:
        patch_results.append({
            "patch_name": "extract_valid_index_none_guard",
            "patch_status": "pattern_missing_no_change",
        })
    main_path.write_text(text, encoding="utf-8")
    status = "patched_runtime_copy" if any(
        row["patch_status"] == "patched_runtime_copy" for row in patch_results
    ) else "already_patched"
    return {
        "patch_name": "sigmark_colab_t2v_runtime_compatibility",
        "patch_status": status,
        "patch_results": patch_results,
        "patched_file": str(main_path),
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
    """按 runtime records 的首次出现顺序构造 SIGMark prompt set。"""

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
        raise KeyError(f"sigmark_prompt_text_missing_for_prompt_ids:{missing}:prompt_suite={prompt_suite_path}")
    if not rows:
        raise RuntimeError("sigmark_no_runtime_prompts_selected")
    return rows


def _write_runtime_prompt_set(
    runtime_source_dir: Path,
    output_root: Path,
    *,
    prompt_set: str,
    runtime_dimension: str,
    prompt_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """把 SSTW prompt suite 转成 SIGMark 官方 prompt_set 目录结构。"""

    prompt_root = runtime_source_dir / "prompt_set"
    prompt_set_dir = prompt_root / f"{prompt_set}_prompt"
    meta_dir = prompt_root / "meta_info"
    if prompt_set_dir.exists():
        _replace_directory(prompt_set_dir, output_root)
    else:
        prompt_set_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_set_dir / f"{runtime_dimension}.txt"
    meta_file = meta_dir / f"{runtime_dimension}.json"
    prompt_texts = [str(row["prompt_text"]).strip() for row in prompt_rows]
    _write_text(prompt_file, "\n".join(prompt_texts) + "\n")
    meta_payload = [
        {
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "sstw_prompt_anchor": "same_prompt_id_from_runtime_detection_records",
        }
        for row in prompt_rows
    ]
    _write_json(meta_file, meta_payload)
    manifest = {
        "manifest_kind": "sigmark_runtime_prompt_set_manifest",
        "prompt_set": prompt_set,
        "runtime_dimension": runtime_dimension,
        "prompt_file": str(prompt_file),
        "meta_file": str(meta_file),
        "prompt_count": len(prompt_rows),
        "prompt_ids": [row["prompt_id"] for row in prompt_rows],
        "prompt_set_policy": "same_prompt_anchor_for_external_baseline_official_hunyuan_flow",
    }
    _write_json(output_root / "sigmark_runtime_prompt_set_manifest.json", manifest)
    return manifest


def _ensure_model_available(config: SigmarkOfficialHunyuanRuntimeConfig, *, output_root: Path) -> dict[str, Any]:
    """检查或下载 HunyuanVideo 官方模型目录。"""

    model_base = Path(config.model_base_path)
    model_dir = model_base / config.model_name
    model_base.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        return {
            "model_status": "ready",
            "model_name": config.model_name,
            "model_base_path": str(model_base),
            "model_dir": str(model_dir),
            "auto_download_hf_model": bool(config.auto_download_hf_model),
        }
    if config.dry_run:
        return {
            "model_status": "dry_run_not_checked",
            "model_name": config.model_name,
            "model_base_path": str(model_base),
            "model_dir": str(model_dir),
            "auto_download_hf_model": bool(config.auto_download_hf_model),
        }
    if not config.auto_download_hf_model:
        raise FileNotFoundError(
            "sigmark_hunyuan_model_missing:"
            f"model_dir={model_dir}:set_SSTW_SIGMARK_AUTO_DOWNLOAD_HF_MODEL=true_or_prepopulate_model_base_path"
        )
    repo_id = HF_MODEL_REPOS.get(config.model_name)
    if not repo_id:
        raise KeyError(f"sigmark_unknown_hf_model_repo:{config.model_name}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - Colab 依赖环境覆盖
        raise ImportError("sigmark_requires_huggingface_hub_for_auto_model_download") from exc
    downloaded = snapshot_download(repo_id=repo_id, local_dir=str(model_dir), local_dir_use_symlinks=False)
    manifest = {
        "model_status": "downloaded",
        "model_name": config.model_name,
        "model_repo_id": repo_id,
        "model_base_path": str(model_base),
        "model_dir": str(model_dir),
        "downloaded_path": str(downloaded),
        "auto_download_hf_model": True,
    }
    _write_json(output_root / "sigmark_model_download_manifest.json", manifest)
    return manifest


def _sigmark_command(config: SigmarkOfficialHunyuanRuntimeConfig, *, runtime_source_dir: Path, mode: str) -> list[str]:
    """构造 SIGMark 官方 main.py 命令。"""

    launcher = [sys.executable, "main.py"]
    if int(config.nproc_per_node) > 1:
        launcher = ["torchrun", f"--nproc_per_node={int(config.nproc_per_node)}", "main.py"]
    return [
        *launcher,
        f"--mode={mode}",
        f"--model_base_path={config.model_base_path}",
        f"--model_name={config.model_name}",
        f"--prompt_set={config.prompt_set}",
        "--watermark_method=sigmark",
        f"--ch_factor={int(config.ch_factor)}",
        f"--hw_factor={int(config.hw_factor)}",
        f"--fr_factor={int(config.fr_factor)}",
        f"--batch_size={int(config.batch_size)}",
        f"--width={int(config.width)}",
        f"--height={int(config.height)}",
        f"--num_frames={int(config.num_frames)}",
        f"--fps={int(config.fps)}",
        f"--guidance_scale={float(config.guidance_scale)}",
        f"--num_steps={int(config.num_inference_steps)}",
        f"--num_videos_per_prompt={int(config.num_videos_per_prompt)}",
        f"--num_videos_per_prompt_diversity={int(config.num_videos_per_prompt_diversity)}",
        "--num_prompts_diversity=1",
        f"--small_scale_test={int(config.small_scale_test)}",
        f"--output_path={config.output_path}",
        f"--precision={normalize_sigmark_precision(config.precision)}",
        f"--quant_text_encoder={int(config.quant_text_encoder)}",
        f"--seed={int(config.seed)}",
    ]


def _run_sigmark_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """运行单条 SIGMark 官方命令并写出 stdout / stderr。"""

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
    )
    stdout_path = log_path.with_name(log_path.name + "_stdout.txt")
    stderr_path = log_path.with_name(log_path.name + "_stderr.txt")
    _write_text(stdout_path, completed.stdout)
    _write_text(stderr_path, completed.stderr)
    return {
        "command": command,
        "cwd": str(cwd),
        "return_code": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _find_bit_accuracy_npz(output_path: Path) -> list[Path]:
    """定位 SIGMark 官方 extract 生成的 bit accuracy npz。"""

    candidates = sorted(output_path.glob("**/*bit_accuracy*.npz"))
    return [path for path in candidates if path.is_file()]


def _score_from_sigmark_bit_accuracy_npz(npz_path: Path) -> dict[str, Any]:
    """从 SIGMark 官方 bit_accuracy npz 中计算聚合 score。"""

    import numpy as np

    payload = np.load(npz_path, allow_pickle=True)
    values: list[float] = []
    npz_keys: list[str] = []
    for key in payload.files:
        array = payload[key]
        npz_keys.append(str(key))
        if getattr(array, "size", 0) == 0:
            continue
        values.extend(float(item) for item in array.reshape(-1))
    if not values:
        raise RuntimeError(f"sigmark_bit_accuracy_npz_empty:{npz_path}")
    mean_score = sum(values) / len(values)
    return {
        "bit_accuracy": round(float(mean_score), 6),
        "external_baseline_score": round(float(mean_score), 6),
        "official_npz_key_count": len(npz_keys),
        "official_npz_value_count": len(values),
        "official_npz_keys_sample": npz_keys[:10],
    }


def _bundle_record_path(bundle_root: Path, record: Mapping[str, Any]) -> Path:
    """构造 SIGMark official bundle 的单条记录路径。"""

    prompt = _safe_token(record.get("prompt_id"))
    seed = _safe_token(record.get("seed_id"))
    attack = _safe_token(record.get("attack_name"))
    return bundle_root / BASELINE_ID / "records" / f"{prompt}__{seed}__{attack}.json"


def write_sigmark_official_bundle_records(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    manifest_path: str | Path,
    bit_accuracy_npz_path: str | Path,
    model_name: str,
    max_records: int | None = None,
) -> dict[str, Any]:
    """把 SIGMark 官方 extract 输出转写为 official bundle records。

    这是适配层, 不是正式论文计分层。它只把官方 bit accuracy 以同一
    prompt / seed / attack 锚点写入项目内 official bundle, 后续统一 runner
    才会生成 `metric_status: measured_formal` 的正式记录。
    """

    root = Path(run_root)
    bundle = Path(bundle_root)
    npz_path = Path(bit_accuracy_npz_path)
    score_payload = _score_from_sigmark_bit_accuracy_npz(npz_path)
    records = _selected_runtime_records(root, max_records)
    generated = 0
    failures: list[dict[str, Any]] = []
    threshold = float(os.environ.get("SSTW_SIGMARK_DETECTION_THRESHOLD", str(DEFAULT_DETECTION_THRESHOLD)))
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
                "external_baseline_official_execution_mode": "sigmark_hunyuan_gen_extract",
                "official_score_assignment_policy": "aggregate_mean_over_sigmark_official_bit_accuracy_npz",
                "attack_protocol_status": "sigmark_official_clean_extract_score_reused_for_runtime_attack_anchor",
                "official_bit_accuracy_npz_path": str(npz_path),
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


def run_sigmark_official_hunyuan_runtime(config: SigmarkOfficialHunyuanRuntimeConfig) -> dict[str, Any]:
    """执行 SIGMark 官方 Hunyuan gen->extract 并生成 official bundle。"""

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
    geometry_manifest = validate_sigmark_watermark_geometry(
        width=config.width,
        height=config.height,
        num_frames=config.num_frames,
        ch_factor=config.ch_factor,
        hw_factor=config.hw_factor,
        fr_factor=config.fr_factor,
    )
    records = _selected_runtime_records(run_root, config.max_records)
    if not records:
        raise RuntimeError(f"sigmark_runtime_detection_records_missing:{run_root / 'records/runtime_detection_records.jsonl'}")
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
    patch_manifest = _patch_sigmark_main_for_t2v(runtime_source_dir)
    prompt_manifest = _write_runtime_prompt_set(
        runtime_source_dir,
        output_root,
        prompt_set=config.prompt_set,
        runtime_dimension=config.runtime_dimension,
        prompt_rows=prompt_rows,
    )
    execution_failure_reason = ""
    try:
        model_manifest = _ensure_model_available(config, output_root=output_root)
    except Exception as exc:
        model_manifest = {
            "model_status": "missing_or_download_failed",
            "model_name": config.model_name,
            "model_base_path": config.model_base_path,
            "auto_download_hf_model": bool(config.auto_download_hf_model),
            "failure_reason": str(exc),
        }
        execution_failure_reason = str(exc)

    gen_command = _sigmark_command(config, runtime_source_dir=runtime_source_dir, mode="gen")
    extract_command = _sigmark_command(config, runtime_source_dir=runtime_source_dir, mode="extract")
    command_results: list[dict[str, Any]] = []
    execution_status = "dry_run_planned" if config.dry_run else "executed"
    if not config.dry_run and not execution_failure_reason:
        gen_result = _run_sigmark_command(
            gen_command,
            cwd=runtime_source_dir,
            log_path=output_root / "logs" / "sigmark_gen",
            timeout_seconds=float(config.timeout_seconds),
        )
        command_results.append(gen_result)
        if int(gen_result["return_code"]) != 0:
            execution_failure_reason = (
                f"sigmark_official_gen_failed:{gen_result['return_code']}:{gen_result['stderr_tail']}"
            )
        if not execution_failure_reason:
            extract_result = _run_sigmark_command(
                extract_command,
                cwd=runtime_source_dir,
                log_path=output_root / "logs" / "sigmark_extract",
                timeout_seconds=float(config.timeout_seconds),
            )
            command_results.append(extract_result)
            if int(extract_result["return_code"]) != 0:
                execution_failure_reason = (
                    f"sigmark_official_extract_failed:{extract_result['return_code']}:{extract_result['stderr_tail']}"
                )

    manifest_path = bundle_root / BASELINE_ID / "official_reference_execution_manifest.json"
    bit_accuracy_candidates = _find_bit_accuracy_npz(output_path) if not config.dry_run and not execution_failure_reason else []
    bundle_result = {
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": 0,
        "failed_bundle_record_count": 0,
        "failures": [],
    }
    selected_bit_accuracy_path = ""
    if bit_accuracy_candidates:
        selected_bit_accuracy_path = str(bit_accuracy_candidates[-1])
        bundle_result = write_sigmark_official_bundle_records(
            run_root=run_root,
            bundle_root=bundle_root,
            manifest_path=manifest_path,
            bit_accuracy_npz_path=selected_bit_accuracy_path,
            model_name=config.model_name,
            max_records=config.max_records,
        )
    elif not config.dry_run and not execution_failure_reason:
        execution_failure_reason = f"sigmark_bit_accuracy_npz_missing:{output_path}"
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
        "official_repository_url": "https://github.com/JeremyZhao1998/SIGMark-release",
        "official_execution_mode": "sigmark_hunyuan_gen_extract",
        "execution_status": execution_status,
        "execution_failure_reason": execution_failure_reason,
        "dry_run": bool(config.dry_run),
        "config": asdict(config),
        "source_audit": source_audit,
        "geometry_manifest": geometry_manifest,
        "patch_manifest": patch_manifest,
        "prompt_manifest": prompt_manifest,
        "model_manifest": model_manifest,
        "gen_command": gen_command,
        "extract_command": extract_command,
        "command_results": command_results,
        "bit_accuracy_npz_candidates": [str(path) for path in bit_accuracy_candidates],
        "selected_bit_accuracy_npz_path": selected_bit_accuracy_path,
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": int(bundle_result["generated_bundle_record_count"]),
        "failed_bundle_record_count": int(bundle_result["failed_bundle_record_count"]),
        "failures": bundle_result.get("failures", []),
        "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
    }
    _write_json(manifest_path, manifest)
    _write_json(output_root / "sigmark_official_hunyuan_execution_manifest.json", manifest)
    if selected_bit_accuracy_path:
        os.environ["SSTW_SIGMARK_BIT_ACCURACY_NPZ"] = selected_bit_accuracy_path
    os.environ["SSTW_SIGMARK_OUTPUT_DIR"] = str(output_path)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行入口参数。"""

    parser = argparse.ArgumentParser(description="运行 SIGMark 官方 Hunyuan gen->extract 并生成 official bundle")
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
    config = build_default_sigmark_official_hunyuan_config_from_env(
        run_root=args.run_root,
        bundle_root=args.bundle_root,
        source_dir=args.source_dir,
        repo_root=args.repo_root,
        resource_root=args.resource_root or None,
        max_records=args.max_records,
    )
    if args.dry_run:
        config = SigmarkOfficialHunyuanRuntimeConfig(**{**asdict(config), "dry_run": True})
    result = run_sigmark_official_hunyuan_runtime(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI 入口由 Notebook 调用
    main()
