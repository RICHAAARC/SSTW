"""VideoShield 官方生成 / 反演流程的项目内运行器。

该模块把第三方 VideoShield 官方仓库纳入 SSTW 的自包含 external baseline
闭环: clone / build / run / adapt / record。它不直接写论文正式
`measured_formal` records, 而是生成 official bundle 和执行 manifest, 后续仍由
统一 `external_baseline_runner` 转写正式记录。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Iterator, Mapping

from external_baseline.official_eval_adapters.common import REPOSITORY_GENERATED_OFFICIAL_PROVENANCE
from external_baseline.runtime_trace_io import build_comparison_unit_id, comparable_detection_records
from external_baseline.score_semantics import official_score_formal_comparison_summary
from external_baseline.video_tensor_io import read_video_tchw_uint8, write_video_tchw


BASELINE_ID = "videoshield"
DEFAULT_MODEL_NAME = "modelscope"
DEFAULT_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
DEFAULT_NUM_FRAMES = 16
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_NUM_INFERENCE_STEPS = 25
DEFAULT_FPS = 8
DEFAULT_CHANNEL_COPY = 1
DEFAULT_FRAMES_COPY = 8
DEFAULT_HW_COPY = 4
DEFAULT_DETECTION_THRESHOLD = 0.55
DEFAULT_POSITIVE_CONTROL_THRESHOLD = 0.5
REQUIRED_SOURCE_FILES = (
    "watermark.py",
    "utils.py",
    "watermark_embedding_and_extraction.py",
    "temporal_tamper_localization.py",
)
OFFICIAL_MODULE_NAMES = (
    "watermark",
    "utils",
    "watermark_embedding_and_extraction",
    "temporal_tamper_localization",
)


@dataclass(frozen=True)
class VideoShieldOfficialRuntimeConfig:
    """VideoShield 官方运行器的显式配置。

    VideoShield 是生成过程中嵌入 latent watermark 的方法。正式 baseline 不能只把
    SSTW / Wan 视频送入 VideoShield detector, 而必须先用 VideoShield 官方生成
    流程得到自己的 watermarked 视频, 再按相同 prompt / seed / attack 协议产生
    official bundle。
    """

    run_root: str
    bundle_root: str
    source_dir: str
    output_root: str
    resource_root: str
    prompt_suite_path: str
    repo_root: str = "."
    model_name: str = DEFAULT_MODEL_NAME
    model_path: str = DEFAULT_MODEL_ID
    max_records: int | None = None
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_frames: int = DEFAULT_NUM_FRAMES
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    num_inversion_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    fps: int = DEFAULT_FPS
    channel_copy: int = DEFAULT_CHANNEL_COPY
    frames_copy: int = DEFAULT_FRAMES_COPY
    hw_copy: int = DEFAULT_HW_COPY
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD
    positive_control_threshold: float = DEFAULT_POSITIVE_CONTROL_THRESHOLD
    device: str = "cuda:0"
    dry_run: bool = False
    allow_prompt_id_fallback: bool = False
    allow_seed_id_fallback: bool = False
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
    """写出稳定 JSON artifact。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_token(value: Any) -> str:
    """把 prompt、seed、attack 或 trace id 转换为路径安全 token。"""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return text.strip("_") or "unknown"


def _stable_seed_from_id(seed_id: str) -> int:
    """从 seed_id 生成稳定整数种子, 仅在显式允许 fallback 时使用。"""

    digest = hashlib.sha256(seed_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % 2_147_483_647


def _is_relative_to(child: Path, parent: Path) -> bool:
    """兼容不同 Python 版本的路径包含关系判断。"""

    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _replace_directory(path: Path, allowed_root: Path) -> None:
    """删除并重建受控目录, 避免误删用户文件。"""

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


def build_default_videoshield_official_config_from_env(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    source_dir: str | Path,
    repo_root: str | Path = ".",
    resource_root: str | Path | None = None,
    max_records: int | None = None,
) -> VideoShieldOfficialRuntimeConfig:
    """从 Colab 环境变量构造 VideoShield 官方运行配置。"""

    root = Path(run_root)
    resources = Path(resource_root) if resource_root else _default_resource_root(root)
    prompt_suite_path = os.environ.get("SSTW_VIDEOSHIELD_PROMPT_SUITE_PATH", "").strip()
    if not prompt_suite_path:
        prompt_suite_path = str(_default_prompt_suite_path(root))
    output_root = os.environ.get("SSTW_VIDEOSHIELD_OFFICIAL_RUNTIME_OUTPUT_ROOT", "").strip()
    if not output_root:
        output_root = str(Path(bundle_root) / BASELINE_ID / "official_runtime")
    max_records_text = os.environ.get("SSTW_VIDEOSHIELD_REFERENCE_MAX_RECORDS", "").strip()
    effective_max_records = int(max_records_text) if max_records_text else max_records
    model_name = os.environ.get("SSTW_VIDEOSHIELD_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    default_model_path = DEFAULT_MODEL_ID if model_name == "modelscope" else "stabilityai/stable-video-diffusion-img2vid-xt"
    model_path = os.environ.get("SSTW_VIDEOSHIELD_MODEL_PATH", default_model_path).strip() or default_model_path
    return VideoShieldOfficialRuntimeConfig(
        run_root=str(root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=output_root,
        resource_root=str(resources),
        prompt_suite_path=prompt_suite_path,
        repo_root=str(repo_root),
        model_name=model_name,
        model_path=model_path,
        max_records=effective_max_records,
        height=_env_int("SSTW_VIDEOSHIELD_HEIGHT", DEFAULT_HEIGHT),
        width=_env_int("SSTW_VIDEOSHIELD_WIDTH", DEFAULT_WIDTH),
        num_frames=_env_int("SSTW_VIDEOSHIELD_NUM_FRAMES", DEFAULT_NUM_FRAMES),
        num_inference_steps=_env_int("SSTW_VIDEOSHIELD_NUM_INFERENCE_STEPS", DEFAULT_NUM_INFERENCE_STEPS),
        num_inversion_steps=_env_int("SSTW_VIDEOSHIELD_NUM_INVERSION_STEPS", DEFAULT_NUM_INFERENCE_STEPS),
        fps=_env_int("SSTW_VIDEOSHIELD_FPS", DEFAULT_FPS),
        channel_copy=_env_int("SSTW_VIDEOSHIELD_CHANNEL_COPY", DEFAULT_CHANNEL_COPY),
        frames_copy=_env_int("SSTW_VIDEOSHIELD_FRAMES_COPY", DEFAULT_FRAMES_COPY),
        hw_copy=_env_int("SSTW_VIDEOSHIELD_HW_COPY", DEFAULT_HW_COPY),
        detection_threshold=_env_float("SSTW_VIDEOSHIELD_DETECTION_THRESHOLD", DEFAULT_DETECTION_THRESHOLD),
        positive_control_threshold=_env_float("SSTW_VIDEOSHIELD_POSITIVE_CONTROL_THRESHOLD", DEFAULT_POSITIVE_CONTROL_THRESHOLD),
        device=os.environ.get("SSTW_VIDEOSHIELD_DEVICE", "cuda:0").strip() or "cuda:0",
        dry_run=_env_bool("SSTW_VIDEOSHIELD_OFFICIAL_DRY_RUN", False),
        allow_prompt_id_fallback=_env_bool("SSTW_VIDEOSHIELD_ALLOW_PROMPT_ID_FALLBACK", False),
        allow_seed_id_fallback=_env_bool("SSTW_VIDEOSHIELD_ALLOW_SEED_ID_FALLBACK", False),
        force_rebuild_runtime_source=_env_bool("SSTW_VIDEOSHIELD_FORCE_REBUILD_RUNTIME_SOURCE", True),
    )


def _ensure_source_ready(source_dir: Path) -> dict[str, Any]:
    """验证 VideoShield 官方仓库最小入口文件。"""

    rows = [{"relative_path": item, "path": str(source_dir / item), "exists": (source_dir / item).exists()} for item in REQUIRED_SOURCE_FILES]
    missing = [row["relative_path"] for row in rows if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"videoshield_official_source_required_files_missing:{missing}:source_dir={source_dir}")
    return {
        "official_source_dir": str(source_dir),
        "required_source_files": rows,
        "source_status": "ready",
    }


def _copy_official_source_to_runtime(source_dir: Path, runtime_source_dir: Path, output_root: Path, *, force: bool) -> dict[str, Any]:
    """把官方源码复制到本次运行目录, 避免直接污染 clone 目录。"""

    if force:
        _replace_directory(runtime_source_dir, output_root)
    elif not runtime_source_dir.exists():
        runtime_source_dir.mkdir(parents=True, exist_ok=True)
    if force or not any(runtime_source_dir.iterdir()):
        ignore = shutil.ignore_patterns(".git", "__pycache__", "results", "*.pyc")
        shutil.copytree(source_dir, runtime_source_dir, dirs_exist_ok=True, ignore=ignore)
    return {
        "runtime_source_dir": str(runtime_source_dir),
        "source_copy_status": "copied" if force else "reused_or_copied",
        "force_rebuild_runtime_source": bool(force),
    }


@contextmanager
def _official_source_context(source_dir: Path) -> Iterator[None]:
    """临时把官方源码目录置于 sys.path 首位并清理同名模块缓存。"""

    resolved = str(source_dir.resolve())
    previous_modules = {name: sys.modules.get(name) for name in OFFICIAL_MODULE_NAMES}
    for name in OFFICIAL_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, resolved)
    previous_cwd = Path.cwd()
    os.chdir(source_dir)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path = [item for item in sys.path if item != resolved]
        for name in OFFICIAL_MODULE_NAMES:
            sys.modules.pop(name, None)
            if previous_modules.get(name) is not None:
                sys.modules[name] = previous_modules[name]  # type: ignore[assignment]


def _prompt_rows_from_suite(prompt_suite_path: Path) -> dict[str, str]:
    """读取 prompt_id 到 prompt_text 映射。"""

    suite = _read_json(prompt_suite_path)
    prompts = suite.get("prompts", [])
    if not isinstance(prompts, list):
        raise TypeError(f"prompt_suite_prompts_must_be_list:{prompt_suite_path}")
    rows: dict[str, str] = {}
    for item in prompts:
        if isinstance(item, Mapping) and item.get("prompt_id") and item.get("prompt_text"):
            rows[str(item["prompt_id"])] = str(item["prompt_text"])
    return rows


def _seed_values_from_suite(prompt_suite_path: Path) -> dict[str, int]:
    """读取 seed_id 到 seed_value 映射。"""

    suite = _read_json(prompt_suite_path)
    rows: dict[str, int] = {}
    for item in suite.get("seeds", []):
        if isinstance(item, Mapping) and item.get("seed_id") and item.get("seed_value") is not None:
            rows[str(item["seed_id"])] = int(item["seed_value"])
    return rows


def _selected_runtime_records(run_root: Path, max_records: int | None) -> list[dict[str, Any]]:
    """读取本次要锚定的 runtime detection records。"""

    records = comparable_detection_records(run_root)
    return records[: int(max_records)] if max_records is not None else records


def _runtime_units(
    records: list[Mapping[str, Any]],
    *,
    prompt_suite_path: Path,
    allow_prompt_id_fallback: bool,
    allow_seed_id_fallback: bool,
) -> list[dict[str, Any]]:
    """按 prompt / seed 首次出现顺序构造 VideoShield 生成单元。"""

    prompt_text_by_id = _prompt_rows_from_suite(prompt_suite_path)
    seed_value_by_id = _seed_values_from_suite(prompt_suite_path)
    units: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    missing_prompts: list[str] = []
    missing_seeds: list[str] = []
    for record in records:
        prompt_id = str(record.get("prompt_id") or "").strip()
        seed_id = str(record.get("seed_id") or "").strip()
        if not prompt_id or not seed_id or (prompt_id, seed_id) in seen:
            continue
        prompt_text = prompt_text_by_id.get(prompt_id)
        if not prompt_text and allow_prompt_id_fallback:
            prompt_text = prompt_id
        if not prompt_text:
            missing_prompts.append(prompt_id)
            continue
        seed_value = seed_value_by_id.get(seed_id)
        if seed_value is None and allow_seed_id_fallback:
            seed_value = _stable_seed_from_id(seed_id)
        if seed_value is None:
            missing_seeds.append(seed_id)
            continue
        seen.add((prompt_id, seed_id))
        units.append({"prompt_id": prompt_id, "prompt_text": prompt_text, "seed_id": seed_id, "seed_value": int(seed_value)})
    if missing_prompts:
        raise KeyError(f"videoshield_prompt_text_missing_for_prompt_ids:{missing_prompts}:prompt_suite={prompt_suite_path}")
    if missing_seeds:
        raise KeyError(f"videoshield_seed_value_missing_for_seed_ids:{missing_seeds}:prompt_suite={prompt_suite_path}")
    if not units:
        raise RuntimeError("videoshield_no_runtime_prompt_seed_units_selected")
    return units


def _write_unit_plan(output_root: Path, units: list[Mapping[str, Any]], records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """写出 prompt / seed / attack 锚点计划, 供 dry-run 和正式运行审计。"""

    plan = {
        "manifest_kind": "videoshield_runtime_unit_plan",
        "baseline_id": BASELINE_ID,
        "unit_count": len(units),
        "runtime_detection_record_count": len(records),
        "units": list(units),
        "comparison_unit_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
        "claim_support_status": "official_runtime_plan_not_claim_evidence",
    }
    path = output_root / "videoshield_runtime_unit_plan.json"
    _write_json(path, plan)
    return {**plan, "unit_plan_path": str(path)}


def _set_generation_seed(seed: int) -> None:
    """设置 Python、NumPy 和 torch 的常规随机种子。"""

    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_modelscope_pipeline(config: VideoShieldOfficialRuntimeConfig) -> tuple[Any, Any]:
    """加载 VideoShield README 中的 ModelScope T2V pipeline 与 inverse scheduler。"""

    import torch
    from diffusers import DDIMInverseScheduler, TextToVideoSDPipeline

    pipe = TextToVideoSDPipeline.from_pretrained(config.model_path, torch_dtype=torch.float16).to(config.device)
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(config.model_path, subfolder="scheduler")
    pipe.safety_checker = None
    return pipe, inverse_scheduler


def _normalise_frames(frames: Any) -> list[Any]:
    """把官方 pipeline 输出统一为 RGB float 数组列表。"""

    import numpy as np
    from PIL import Image

    result: list[Any] = []
    for frame in list(frames):
        if isinstance(frame, Image.Image):
            array = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        else:
            array = np.asarray(frame, dtype=np.float32)
            if (float(array.max()) if array.size else 0.0) > 1.5:
                array = array / 255.0
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[-1] >= 4:
                array = array[..., :3]
        result.append(array.clip(0.0, 1.0))
    if not result:
        raise RuntimeError("videoshield_frames_empty")
    return result


def _frames_to_tchw(frames: list[Any]) -> Any:
    """把 RGB float 帧列表转换为 `[T, C, H, W]` 视频张量。"""

    import numpy as np
    import torch

    array = np.stack(frames, axis=0)
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().float()


def _read_video_frames_float(video_path: Path) -> list[Any]:
    """从 mp4 读取 RGB float 帧列表。"""

    tensor, _ = read_video_tchw_uint8(video_path, empty_error="videoshield_attacked_video_empty")
    array = tensor.permute(0, 2, 3, 1).contiguous().numpy().astype("float32") / 255.0
    return [frame for frame in array]


def _video_id(unit: Mapping[str, Any]) -> str:
    """构造不会暴露完整 prompt 文本的视频目录名。"""

    digest = hashlib.sha1(str(unit.get("prompt_text") or "").encode("utf-8")).hexdigest()[:12]
    return f"{_safe_token(unit.get('prompt_id'))}__{_safe_token(unit.get('seed_id'))}__{digest}"


def _create_watermark(config: VideoShieldOfficialRuntimeConfig) -> Any:
    """创建官方 VideoShield watermark 对象。"""

    from watermark import VideoShield

    return VideoShield(
        ch_factor=int(config.channel_copy),
        hw_factor=int(config.hw_copy),
        frame_factor=int(config.frames_copy),
        height=int(config.height / 8),
        width=int(config.width / 8),
        num_frames=int(config.num_frames),
        device=config.device,
    )


def _invert_frames_modelscope(pipe: Any, inverse_scheduler: Any, frames: list[Any], config: VideoShieldOfficialRuntimeConfig) -> Any:
    """调用官方 ModelScope 反演路径把视频帧转回 latent。"""

    from utils import get_video_latents, transform_video

    pipe.scheduler = inverse_scheduler
    video_tensor = transform_video(frames).to(pipe.vae.dtype).to(config.device)
    video_latents = get_video_latents(pipe.vae, video_tensor, sample=False, permute=True)
    return pipe(
        prompt="",
        latents=video_latents,
        num_inference_steps=int(config.num_inversion_steps),
        guidance_scale=1.0,
        output_type="latent",
    ).frames


def _generate_watermarked_unit(
    *,
    unit: Mapping[str, Any],
    unit_root: Path,
    pipe: Any,
    inverse_scheduler: Any,
    scheduler: Any,
    config: VideoShieldOfficialRuntimeConfig,
) -> dict[str, Any]:
    """按官方 VideoShield 生成路径产生 watermarked 视频与同基座 clean negative 视频。

    clean negative 使用同一个 prompt / seed / 生成模型, 但不注入 VideoShield
    watermark latents。后续检测仍使用同一 official watermark template, 用于在
    validation_scale 中校准该 baseline 自身的 clean negative 分布。
    """

    import torch

    if config.model_name != "modelscope":
        raise RuntimeError("videoshield_project_runtime_currently_supports_modelscope_only")
    from utils import save_video_frames

    _set_generation_seed(int(unit["seed_value"]))
    unit_root.mkdir(parents=True, exist_ok=True)
    watermark = _create_watermark(config)
    init_latents_w = watermark.create_watermark_and_return_w()
    wm_info_path = unit_root / "wm_info.bin"
    torch.save(
        {"m": watermark.m, "watermark": watermark.watermark, "key": watermark.key, "nonce": watermark.nonce},
        wm_info_path,
    )
    pipe.scheduler = scheduler
    generated_frames = pipe(
        prompt=str(unit["prompt_text"]),
        latents=init_latents_w,
        num_frames=int(config.num_frames),
        height=int(config.height),
        width=int(config.width),
        num_inference_steps=int(config.num_inference_steps),
        guidance_scale=9.0,
    ).frames[0]
    frames = _normalise_frames(generated_frames)
    watermarked_video_path = unit_root / "wm.mp4"
    write_video_tchw(watermarked_video_path, _frames_to_tchw(frames), fps=float(config.fps))
    frames_dir = unit_root / "wm" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    save_video_frames(frames, str(frames_dir))
    reversed_latents = _invert_frames_modelscope(pipe, inverse_scheduler, frames, config)
    positive_control_score = float(watermark.eval_watermark(reversed_latents))
    _set_generation_seed(int(unit["seed_value"]))
    pipe.scheduler = scheduler
    clean_generated_frames = pipe(
        prompt=str(unit["prompt_text"]),
        num_frames=int(config.num_frames),
        height=int(config.height),
        width=int(config.width),
        num_inference_steps=int(config.num_inference_steps),
        guidance_scale=9.0,
    ).frames[0]
    clean_frames = _normalise_frames(clean_generated_frames)
    clean_video_path = unit_root / "clean_negative.mp4"
    write_video_tchw(clean_video_path, _frames_to_tchw(clean_frames), fps=float(config.fps))
    clean_reversed_latents = _invert_frames_modelscope(pipe, inverse_scheduler, clean_frames, config)
    clean_negative_score = float(watermark.eval_watermark(clean_reversed_latents))
    return {
        "unit": dict(unit),
        "unit_root": str(unit_root),
        "watermarked_video_path": str(watermarked_video_path),
        "clean_negative_video_path": str(clean_video_path),
        "frames_dir": str(frames_dir),
        "wm_info_path": str(wm_info_path),
        "watermark": watermark,
        "frames": frames,
        "clean_frames": clean_frames,
        "clean_negative_score": clean_negative_score,
        "positive_control_score": positive_control_score,
        "positive_control_status": "PASS" if positive_control_score >= float(config.positive_control_threshold) else "FAIL",
    }


def _apply_runtime_attack_to_frames(
    frames: list[Any],
    *,
    attack_name: str,
    output_video_path: Path,
    fps: int,
) -> tuple[list[Any], dict[str, Any]]:
    """对 VideoShield 自己生成的视频施加 SSTW runtime attack 锚点。"""

    normalized = str(attack_name or "").strip().lower()
    if normalized == "video_compression_runtime":
        attacked_frames = list(frames)
        protocol_status = "video_compression_runtime_decode_reencode"
    elif normalized == "temporal_crop_runtime":
        attacked_frames = frames[1:-1] if len(frames) >= 4 else list(frames)
        protocol_status = "temporal_crop_runtime_drop_first_and_last_frame"
    elif normalized == "frame_rate_resampling_runtime":
        attacked_frames = frames[::2] if len(frames) >= 3 else list(frames)
        protocol_status = "frame_rate_resampling_runtime_keep_every_second_frame"
    else:
        raise ValueError(f"unsupported_videoshield_runtime_attack:{attack_name}")
    video_tensor = _frames_to_tchw(attacked_frames)
    write_info = write_video_tchw(output_video_path, video_tensor, fps=float(fps))
    decoded_frames = _read_video_frames_float(output_video_path)
    return decoded_frames, {
        "attack_protocol_status": "project_runtime_attack_applied_to_videoshield_watermarked_video",
        "attack_transform": protocol_status,
        "attacked_video_path": str(output_video_path),
        "attacked_frame_count_before_decode": len(attacked_frames),
        "attacked_frame_count_after_decode": len(decoded_frames),
        "video_write_info": write_info,
    }


def _detect_attacked_frames(
    *,
    frames: list[Any],
    watermark: Any,
    pipe: Any,
    inverse_scheduler: Any,
    config: VideoShieldOfficialRuntimeConfig,
) -> dict[str, Any]:
    """调用官方反演与 temporal matching 逻辑检测攻击后视频。"""

    reversed_latents = _invert_frames_modelscope(pipe, inverse_scheduler, frames, config)
    reversed_m = (reversed_latents > 0).int()
    template_m = watermark.m.to(config.device)
    b, c, f, h, w = reversed_m.size()
    t_f = template_m.shape[2]
    reversed_m_repeat = reversed_m.permute(2, 0, 1, 3, 4).reshape(f, b * c * h * w).unsqueeze(1).repeat(1, t_f, 1)
    template_m_repeat = template_m.permute(2, 0, 1, 3, 4).reshape(t_f, b * c * h * w).unsqueeze(0).repeat(f, 1, 1)
    cmp_bits = (reversed_m_repeat == template_m_repeat).float().mean(dim=2)
    max_values, pred_frame_ids = cmp_bits.float().max(dim=1)
    pred_mask = max_values < float(config.detection_threshold)
    masked_pred_frame_ids = pred_frame_ids.masked_fill(pred_mask, -1)
    score = float(max_values.float().mean().item()) if max_values.numel() else 0.0
    detected_fraction = float((max_values >= float(config.detection_threshold)).float().mean().item()) if max_values.numel() else 0.0
    return {
        "external_baseline_score": round(score, 6),
        "raw_detector_score": round(score, 6),
        "confidence": round(score, 6),
        "detected": score >= float(config.detection_threshold),
        "threshold": float(config.detection_threshold),
        "score_semantics": "watermark_presence_confidence",
        "score_orientation": "higher_is_more_watermarked",
        "videoshield_temporal_alignment_score": round(score, 6),
        "videoshield_detected_frame_fraction": round(detected_fraction, 6),
        "observed_frame_count": int(f),
        "template_frame_count": int(t_f),
        "videoshield_pred_frame_ids": [int(value) for value in masked_pred_frame_ids.detach().cpu().tolist()],
        "videoshield_max_values_min": round(float(max_values.min().item()), 6) if max_values.numel() else 0.0,
        "videoshield_max_values_mean": round(float(max_values.mean().item()), 6) if max_values.numel() else 0.0,
        "videoshield_max_values_max": round(float(max_values.max().item()), 6) if max_values.numel() else 0.0,
        "official_detection_logic": "videoshield_temporal_tamper_latent_inversion_frame_template_matching",
    }


def _bundle_record_path(bundle_root: Path, record: Mapping[str, Any]) -> Path:
    """构造 VideoShield official bundle 的单条记录路径。"""

    return (
        bundle_root
        / BASELINE_ID
        / "records"
        / f"{_safe_token(record.get('prompt_id'))}__{_safe_token(record.get('seed_id'))}__{_safe_token(record.get('attack_name'))}.json"
    )


def run_videoshield_official_runtime(config: VideoShieldOfficialRuntimeConfig) -> dict[str, Any]:
    """执行 VideoShield 官方生成 -> runtime attack -> 官方反演检测闭环。"""

    run_root = Path(config.run_root)
    bundle_root = Path(config.bundle_root)
    source_dir = Path(config.source_dir)
    output_root = Path(config.output_root)
    prompt_suite_path = Path(config.prompt_suite_path)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    source_audit = _ensure_source_ready(source_dir)
    records = _selected_runtime_records(run_root, config.max_records)
    if not records:
        raise RuntimeError(f"videoshield_runtime_detection_records_missing:{run_root / 'records/runtime_detection_records.jsonl'}")
    units = _runtime_units(
        records,
        prompt_suite_path=prompt_suite_path,
        allow_prompt_id_fallback=config.allow_prompt_id_fallback,
        allow_seed_id_fallback=config.allow_seed_id_fallback,
    )
    runtime_source_dir = output_root / "source_runtime"
    source_copy_audit = _copy_official_source_to_runtime(
        source_dir,
        runtime_source_dir,
        output_root,
        force=bool(config.force_rebuild_runtime_source),
    )
    unit_plan = _write_unit_plan(output_root, units, records)
    manifest_path = bundle_root / BASELINE_ID / "official_reference_execution_manifest.json"
    generated = 0
    failures: list[dict[str, Any]] = []
    generation_rows: dict[tuple[str, str], dict[str, Any]] = {}
    execution_status = "dry_run_planned" if config.dry_run else "executed"
    positive_control_rows: list[dict[str, Any]] = []

    if not config.dry_run:
        with _official_source_context(runtime_source_dir):
            pipe, inverse_scheduler = _load_modelscope_pipeline(config)
            scheduler = pipe.scheduler
            for unit in units:
                try:
                    unit_result = _generate_watermarked_unit(
                        unit=unit,
                        unit_root=output_root / "generated_units" / _video_id(unit),
                        pipe=pipe,
                        inverse_scheduler=inverse_scheduler,
                        scheduler=scheduler,
                        config=config,
                    )
                    positive_control_rows.append({
                        "prompt_id": unit["prompt_id"],
                        "seed_id": unit["seed_id"],
                        "positive_control_score": round(float(unit_result["positive_control_score"]), 6),
                        "positive_control_threshold": float(config.positive_control_threshold),
                        "positive_control_status": unit_result["positive_control_status"],
                    })
                    if unit_result["positive_control_status"] != "PASS":
                        raise RuntimeError(
                            "videoshield_positive_control_failed:"
                            f"prompt_id={unit['prompt_id']}:seed_id={unit['seed_id']}:"
                            f"score={unit_result['positive_control_score']}:threshold={config.positive_control_threshold}"
                        )
                    generation_rows[(str(unit["prompt_id"]), str(unit["seed_id"]))] = unit_result
                except Exception as exc:  # pragma: no cover - 依赖 Colab GPU、HF 模型和官方反演
                    failures.append({
                        "baseline_id": BASELINE_ID,
                        "prompt_id": unit.get("prompt_id"),
                        "seed_id": unit.get("seed_id"),
                        "failure_stage": "generation_or_positive_control",
                        "failure_reason": str(exc),
                    })
            if not failures:
                video_dir = bundle_root / BASELINE_ID / "videos"
                for record in records:
                    output_json = _bundle_record_path(bundle_root, record)
                    try:
                        key = (str(record.get("prompt_id") or ""), str(record.get("seed_id") or ""))
                        unit_result = generation_rows[key]
                        attacked_video_path = video_dir / f"{output_json.stem}_attacked.mp4"
                        attacked_frames, attack_info = _apply_runtime_attack_to_frames(
                            unit_result["frames"],
                            attack_name=str(record.get("attack_name") or ""),
                            output_video_path=attacked_video_path,
                            fps=int(config.fps),
                        )
                        score_payload = _detect_attacked_frames(
                            frames=attacked_frames,
                            watermark=unit_result["watermark"],
                            pipe=pipe,
                            inverse_scheduler=inverse_scheduler,
                            config=config,
                        )
                        clean_negative_video_path = video_dir / f"{output_json.stem}_clean_negative.mp4"
                        clean_negative_frames, clean_negative_attack_info = _apply_runtime_attack_to_frames(
                            unit_result["clean_frames"],
                            attack_name=str(record.get("attack_name") or ""),
                            output_video_path=clean_negative_video_path,
                            fps=int(config.fps),
                        )
                        clean_negative_payload = _detect_attacked_frames(
                            frames=clean_negative_frames,
                            watermark=unit_result["watermark"],
                            pipe=pipe,
                            inverse_scheduler=inverse_scheduler,
                            config=config,
                        )
                        payload = {
                            **score_payload,
                            "external_baseline_clean_negative_score": clean_negative_payload["raw_detector_score"],
                            "external_baseline_clean_negative_score_semantics": clean_negative_payload["score_semantics"],
                            "external_baseline_clean_negative_video_path": clean_negative_attack_info["attacked_video_path"],
                            "official_result_provenance": REPOSITORY_GENERATED_OFFICIAL_PROVENANCE,
                            "official_adapter_baseline_id": BASELINE_ID,
                            "official_baseline_id": BASELINE_ID,
                            "official_source_dir": str(source_dir),
                            "official_runtime_source_dir": str(runtime_source_dir),
                            "external_baseline_generation_model_id": f"videoshield_{config.model_name}",
                            "external_baseline_official_execution_mode": (
                                "videoshield_generation_watermarked_video_project_runtime_attack_official_inversion"
                            ),
                            "official_score_extraction_policy": (
                                "videoshield_official_latent_inversion_template_matching_confidence"
                            ),
                            "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                            "attack_protocol_status": attack_info["attack_protocol_status"],
                            "attack_transform": attack_info["attack_transform"],
                            "official_watermark_entropy_policy": "official_chacha20_key_nonce_recorded_in_wm_info_bin",
                            "external_baseline_source_video_path": unit_result["watermarked_video_path"],
                            "external_baseline_attacked_video_path": attack_info["attacked_video_path"],
                            "official_clean_negative_source_video_path": unit_result["clean_negative_video_path"],
                            "official_wm_info_path": unit_result["wm_info_path"],
                            "official_frames_dir": unit_result["frames_dir"],
                            "source_sstw_video_path": str(record.get("source_video_path") or ""),
                            "sstw_attacked_video_path": str(record.get("attacked_video_path") or ""),
                            "runtime_comparison_unit_id": build_comparison_unit_id(BASELINE_ID, record),
                            "attack_name": record.get("attack_name"),
                            "prompt_id": record.get("prompt_id"),
                            "seed_id": record.get("seed_id"),
                            "trajectory_trace_id": record.get("trajectory_trace_id"),
                            "official_execution_manifest_path": str(manifest_path),
                            "claim_support_status": "official_reference_bundle_written_not_claim_by_itself",
                        }
                        payload = {
                            **payload,
                            **official_score_formal_comparison_summary(payload),
                            **official_score_formal_comparison_summary(payload, clean_negative=True),
                        }
                        _write_json(output_json, payload)
                        generated += 1
                    except Exception as exc:  # pragma: no cover - 依赖 Colab GPU、视频编码和官方反演
                        failures.append({
                            "baseline_id": BASELINE_ID,
                            "prompt_id": record.get("prompt_id"),
                            "seed_id": record.get("seed_id"),
                            "attack_name": record.get("attack_name"),
                            "failure_stage": "attack_or_detection_or_bundle_write",
                            "failure_reason": str(exc),
                        })

    manifest = {
        "manifest_kind": "videoshield_official_reference_execution_manifest",
        "baseline_id": BASELINE_ID,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "official_repository_url": "https://github.com/hurunyi/VideoShield",
        "official_source_dir": str(source_dir),
        "official_runtime_source_dir": str(runtime_source_dir),
        "execution_status": execution_status,
        "dry_run": bool(config.dry_run),
        "config": asdict(config),
        "source_audit": source_audit,
        "source_copy_audit": source_copy_audit,
        "unit_plan": unit_plan,
        "positive_control_rows": positive_control_rows,
        "input_runtime_detection_record_count": len(records),
        "generated_video_unit_count": len(units),
        "generated_bundle_record_count": generated,
        "failed_bundle_record_count": len(failures),
        "failures": failures[:20],
        "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
    }
    _write_json(manifest_path, manifest)
    _write_json(bundle_root / BASELINE_ID / "videoshield_official_runtime_execution_manifest.json", manifest)
    return manifest


def main() -> None:
    """CLI 入口。"""

    parser = argparse.ArgumentParser(description="运行 VideoShield 官方生成 / 反演 official bundle。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--source-dir", default="external_baseline/primary/videoshield/source")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--resource-root", default="")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = build_default_videoshield_official_config_from_env(
        run_root=args.run_root,
        bundle_root=args.bundle_root,
        source_dir=args.source_dir,
        repo_root=args.repo_root,
        resource_root=args.resource_root or None,
        max_records=args.max_records,
    )
    if args.dry_run:
        config = VideoShieldOfficialRuntimeConfig(**{**asdict(config), "dry_run": True})
    result = run_videoshield_official_runtime(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI 入口由 Notebook 调用
    main()
