"""VidSig 官方 Video-Signature 生成 / 攻击 / 检测流程运行器。

该模块把 VidSig 官方仓库纳入 SSTW 的自包含 external baseline 闭环:
clone / build / run / adapt / record。它不直接写论文正式 `measured_formal`
records, 而是生成 official bundle 和执行 manifest, 后续仍由统一 runner 转写。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
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
from external_baseline.score_semantics import official_score_formal_comparison_summary


BASELINE_ID = "vidsig"
DEFAULT_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_NUM_FRAMES = 16
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_FPS = 8
DEFAULT_KEY = "100011100001001101101100100011111101111110000000"
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_POSITIVE_CONTROL_BIT_ACCURACY_THRESHOLD = 0.5


@dataclass(frozen=True)
class VidSigOfficialRuntimeConfig:
    """VidSig 官方运行器的显式配置。

    VidSig 是生成过程中嵌入签名的视频水印方法, 因此正式 baseline 不能只拿
    SSTW / Wan 视频调用 detector。正确流程必须先用 VidSig 官方 `generate_ms.py`
    生成 clean / watermarked 视频, 再对 VidSig 自己的 watermarked 视频施加同一
    runtime attack 协议, 最后调用官方 `attack.py` 检测。
    """

    run_root: str
    bundle_root: str
    source_dir: str
    output_root: str
    resource_root: str
    prompt_suite_path: str
    repo_root: str = "."
    model_id: str = DEFAULT_MODEL_ID
    msg_decoder_path: str = ""
    vae_checkpoint_path: str = ""
    max_records: int | None = None
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_frames: int = DEFAULT_NUM_FRAMES
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    fps: int = DEFAULT_FPS
    key: str = DEFAULT_KEY
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD
    positive_control_bit_accuracy_threshold: float = DEFAULT_POSITIVE_CONTROL_BIT_ACCURACY_THRESHOLD
    dry_run: bool = False
    timeout_seconds: float = 0.0
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


def _default_resource_file(resource_root: Path, relative_candidates: tuple[str, ...], env_name: str) -> str:
    """解析默认官方资源文件路径。"""

    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    for relative_path in relative_candidates:
        candidate = resource_root / relative_path
        if candidate.exists():
            return str(candidate)
    return ""


def build_default_vidsig_official_config_from_env(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    source_dir: str | Path,
    repo_root: str | Path = ".",
    resource_root: str | Path | None = None,
    max_records: int | None = None,
) -> VidSigOfficialRuntimeConfig:
    """从 Colab 环境变量构造 VidSig 官方运行配置。"""

    root = Path(run_root)
    resources = Path(resource_root) if resource_root else _default_resource_root(root)
    prompt_suite_path = os.environ.get("SSTW_VIDSIG_PROMPT_SUITE_PATH", "").strip()
    if not prompt_suite_path:
        prompt_suite_path = str(_default_prompt_suite_path(root))
    output_root = os.environ.get("SSTW_VIDSIG_OFFICIAL_RUNTIME_OUTPUT_ROOT", "").strip()
    if not output_root:
        output_root = str(Path(bundle_root) / BASELINE_ID / "official_runtime")
    max_records_text = os.environ.get("SSTW_VIDSIG_REFERENCE_MAX_RECORDS", "").strip()
    effective_max_records = int(max_records_text) if max_records_text else max_records
    msg_decoder_path = _default_resource_file(
        resources,
        ("vidsig/ckpts/msg_decoder/dec_48b_whit.torchscript.pt", "vidsig/dec_48b_whit.torchscript.pt"),
        "SSTW_VIDSIG_MSG_DECODER_PATH",
    )
    vae_checkpoint_path = _default_resource_file(
        resources,
        ("vidsig/ckpts/vae/modelscope/checkpoint.pth", "vidsig/ckpts/vae/svd/checkpoint.pth", "vidsig/checkpoint.pth"),
        "SSTW_VIDSIG_VAE_CHECKPOINT_PATH",
    )
    return VidSigOfficialRuntimeConfig(
        run_root=str(root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=output_root,
        resource_root=str(resources),
        prompt_suite_path=prompt_suite_path,
        repo_root=str(repo_root),
        model_id=os.environ.get("SSTW_VIDSIG_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID,
        msg_decoder_path=msg_decoder_path,
        vae_checkpoint_path=vae_checkpoint_path,
        max_records=effective_max_records,
        height=_env_int("SSTW_VIDSIG_HEIGHT", DEFAULT_HEIGHT),
        width=_env_int("SSTW_VIDSIG_WIDTH", DEFAULT_WIDTH),
        num_frames=_env_int("SSTW_VIDSIG_NUM_FRAMES", DEFAULT_NUM_FRAMES),
        num_inference_steps=_env_int("SSTW_VIDSIG_NUM_INFERENCE_STEPS", DEFAULT_NUM_INFERENCE_STEPS),
        fps=_env_int("SSTW_VIDSIG_FPS", DEFAULT_FPS),
        key=os.environ.get("SSTW_VIDSIG_KEY", DEFAULT_KEY).strip() or DEFAULT_KEY,
        detection_threshold=_env_float("SSTW_VIDSIG_DETECTION_THRESHOLD", DEFAULT_DETECTION_THRESHOLD),
        positive_control_bit_accuracy_threshold=_env_float(
            "SSTW_VIDSIG_POSITIVE_CONTROL_BIT_ACCURACY_THRESHOLD",
            DEFAULT_POSITIVE_CONTROL_BIT_ACCURACY_THRESHOLD,
        ),
        dry_run=_env_bool("SSTW_VIDSIG_OFFICIAL_DRY_RUN", False),
        timeout_seconds=_env_float("SSTW_VIDSIG_OFFICIAL_TIMEOUT_SECONDS", 0.0),
        allow_prompt_id_fallback=_env_bool("SSTW_VIDSIG_ALLOW_PROMPT_ID_FALLBACK", False),
        allow_seed_id_fallback=_env_bool("SSTW_VIDSIG_ALLOW_SEED_ID_FALLBACK", False),
        force_rebuild_runtime_source=_env_bool("SSTW_VIDSIG_FORCE_REBUILD_RUNTIME_SOURCE", True),
    )


def _ensure_source_ready(source_dir: Path) -> dict[str, Any]:
    """验证官方源码目录。"""

    required = ("src/generate_ms.py", "src/attack.py", "yamls/generate_ms.yml")
    rows = [{"relative_path": item, "path": str(source_dir / item), "exists": (source_dir / item).exists()} for item in required]
    missing = [row["relative_path"] for row in rows if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"vidsig_official_source_required_files_missing:{missing}:source_dir={source_dir}")
    return {"official_source_dir": str(source_dir), "required_source_files": rows, "source_status": "ready"}


def _ensure_resource_file(path: str, *, env_name: str, role: str) -> dict[str, Any]:
    """验证官方 checkpoint 资源文件。"""

    resource_path = Path(path) if path else Path()
    exists = bool(path) and resource_path.exists() and resource_path.is_file()
    if not exists:
        raise FileNotFoundError(f"vidsig_required_resource_missing:{env_name}:{role}:{path or 'empty'}")
    return {"env_var": env_name, "resource_role": role, "resource_path": str(resource_path), "resource_exists": True}


def _copy_official_source_to_runtime(source_dir: Path, runtime_source_dir: Path, output_root: Path, *, force: bool) -> None:
    """复制官方源码到 runtime 工作副本。"""

    if runtime_source_dir.exists() and force:
        _replace_directory(runtime_source_dir, output_root)
        runtime_source_dir.rmdir()
    if runtime_source_dir.exists():
        return

    def _ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {".git", "__pycache__", ".pytest_cache", "output", "outputs", "logs"}
        return {name for name in names if name in ignored or name.endswith(".pyc")}

    runtime_source_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, runtime_source_dir, ignore=_ignore)


def _patch_generate_ms_for_colab(runtime_source_dir: Path) -> dict[str, Any]:
    """修补 runtime 副本中的 VidSig `generate_ms.py` Colab 兼容问题。

    官方脚本在当前 diffusers 版本下会得到 PIL frame list, 不能直接传给
    TorchScript decoder。该修补只作用于本次 runtime 工作副本, 不改写第三方
    source snapshot, 目的是让 positive control 能真实验证 VidSig 自己生成的
    watermarked video 是否可被官方 decoder 读出。
    """

    generate_path = runtime_source_dir / "src" / "generate_ms.py"
    if not generate_path.exists():
        return {"patch_status": "source_missing", "generate_ms_path": str(generate_path)}
    text = generate_path.read_text(encoding="utf-8")
    marker = "def sstw_prepare_vidsig_decoder_input"
    patched_steps: list[str] = []
    if marker not in text:
        helper = '''

def sstw_prepare_vidsig_decoder_input(frames, device):
    tensors = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().float()
            if tensor.ndim == 3 and tensor.shape[0] not in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0
        else:
            tensor = transforms.ToTensor()(frame).float()
        tensor = tensor[:3]
        tensors.append(normalize(tensor))
    return torch.stack(tensors).to(device)
'''
        anchor = "normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  \n                                std=[0.229, 0.224, 0.225])\n"
        if anchor in text:
            text = text.replace(anchor, anchor + helper, 1)
            patched_steps.append("insert_decoder_input_helper")
    if "    msg_decoder.eval()\n    msg_decoder = msg_decoder.to(device)\n" not in text:
        before = "    msg_decoder.eval()\n"
        if before in text:
            text = text.replace(before, before + "    msg_decoder = msg_decoder.to(device)\n", 1)
            patched_steps.append("move_msg_decoder_to_device")
    before_decode = "            decoded = msg_decoder(w_frames)\n"
    after_decode = "            decoded_input = sstw_prepare_vidsig_decoder_input(w_frames, device)\n            decoded = msg_decoder(decoded_input)\n"
    if before_decode in text:
        text = text.replace(before_decode, after_decode, 1)
        patched_steps.append("convert_diffusers_frames_before_decoder")
    generate_path.write_text(text, encoding="utf-8")
    patch_status = "patched" if patched_steps else "no_matching_patch_points"
    return {"patch_status": patch_status, "generate_ms_path": str(generate_path), "patched_steps": patched_steps}


def _prompt_rows_from_suite(prompt_suite_path: Path) -> dict[str, str]:
    """读取 prompt_id 到 prompt_text 映射。"""

    suite = _read_json(prompt_suite_path)
    rows: dict[str, str] = {}
    for item in suite.get("prompts", []):
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


def _runtime_prompt_seed_rows(
    records: list[Mapping[str, Any]],
    *,
    prompt_suite_path: Path,
    allow_prompt_id_fallback: bool,
    allow_seed_id_fallback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 runtime records 首次出现顺序构造 VidSig prompt / seed 集合。"""

    prompt_text_by_id = _prompt_rows_from_suite(prompt_suite_path)
    seed_value_by_id = _seed_values_from_suite(prompt_suite_path)
    prompt_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_seeds: set[str] = set()
    missing_prompts: list[str] = []
    missing_seeds: list[str] = []
    for record in records:
        prompt_id = str(record.get("prompt_id") or "").strip()
        if prompt_id and prompt_id not in seen_prompts:
            prompt_text = prompt_text_by_id.get(prompt_id)
            if not prompt_text and allow_prompt_id_fallback:
                prompt_text = prompt_id
            if not prompt_text:
                missing_prompts.append(prompt_id)
            else:
                seen_prompts.add(prompt_id)
                prompt_rows.append({"prompt_id": prompt_id, "prompt_text": prompt_text})
        seed_id = str(record.get("seed_id") or "").strip()
        if seed_id and seed_id not in seen_seeds:
            seed_value = seed_value_by_id.get(seed_id)
            if seed_value is None and allow_seed_id_fallback:
                seed_value = _stable_seed_from_id(seed_id)
            if seed_value is None:
                missing_seeds.append(seed_id)
            else:
                seen_seeds.add(seed_id)
                seed_rows.append({"seed_id": seed_id, "seed_value": int(seed_value)})
    if missing_prompts:
        raise KeyError(f"vidsig_prompt_text_missing_for_prompt_ids:{missing_prompts}:prompt_suite={prompt_suite_path}")
    if missing_seeds:
        raise KeyError(f"vidsig_seed_value_missing_for_seed_ids:{missing_seeds}:prompt_suite={prompt_suite_path}")
    if not prompt_rows:
        raise RuntimeError("vidsig_no_runtime_prompts_selected")
    if not seed_rows:
        raise RuntimeError("vidsig_no_runtime_seeds_selected")
    return prompt_rows, seed_rows


def _json_scalar(value: Any) -> str:
    """把 YAML 标量写成 JSON 兼容格式。"""

    return json.dumps(value, ensure_ascii=False)


def _write_vidsig_generation_inputs(
    *,
    runtime_source_dir: Path,
    output_root: Path,
    config: VidSigOfficialRuntimeConfig,
    prompt_rows: list[Mapping[str, Any]],
    seed_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """把 SSTW prompt / seed 转成 VidSig 官方 `generate_ms.yml`。"""

    prompt_file = runtime_source_dir / "sstw_prompt.txt"
    _write_text(prompt_file, "\n".join(str(row["prompt_text"]).strip() for row in prompt_rows) + "\n")
    official_output_root = output_root / "official_generate_ms_outputs"
    clean_dir = official_output_root / "original"
    watermarked_dir = official_output_root / "video_signature"
    log_dir = official_output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = runtime_source_dir / "yamls" / "generate_ms.yml"
    yaml_lines = [
        f"prompt_file: {_json_scalar(str(prompt_file))}",
        f"image_path: {_json_scalar(str(runtime_source_dir / 'SourceImages'))}",
        f"msg_decoder_path: {_json_scalar(config.msg_decoder_path)}",
        f"ckpt_path: {_json_scalar(config.vae_checkpoint_path)}",
        f"model_id: {_json_scalar(config.model_id)}",
        f"height: {int(config.height)}",
        f"width: {int(config.width)}",
        f"num_frames: {int(config.num_frames)}",
        f"num_inference_steps: {int(config.num_inference_steps)}",
        f"nw_saved_dir: {_json_scalar(str(clean_dir))}",
        f"w_saved_dir: {_json_scalar(str(watermarked_dir))}",
        f"output_dir: {_json_scalar(str(log_dir))}",
        f"key: {_json_scalar(config.key)}",
        "seed:",
        *[f"  - {int(row['seed_value'])}" for row in seed_rows],
    ]
    _write_text(yaml_path, "\n".join(yaml_lines) + "\n")
    manifest = {
        "manifest_kind": "vidsig_runtime_prompt_seed_manifest",
        "prompt_file": str(prompt_file),
        "yaml_path": str(yaml_path),
        "official_output_root": str(official_output_root),
        "clean_video_dir": str(clean_dir / "videos"),
        "clean_frame_dir": str(clean_dir / "frames"),
        "watermarked_video_dir": str(watermarked_dir / "videos"),
        "watermarked_frame_dir": str(watermarked_dir / "frames"),
        "generation_log_path": str(log_dir / "log.txt"),
        "prompt_count": len(prompt_rows),
        "seed_count": len(seed_rows),
        "prompt_rows": list(prompt_rows),
        "seed_rows": list(seed_rows),
        "prompt_seed_policy": "same_prompt_seed_anchor_for_vidsig_official_generate_ms_flow",
    }
    _write_json(output_root / "vidsig_runtime_prompt_seed_manifest.json", manifest)
    return manifest


def _run_command(command: list[str], *, cwd: Path, log_prefix: Path, timeout_seconds: float) -> dict[str, Any]:
    """运行一条官方命令并写出 stdout / stderr。"""

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
    )
    stdout_path = log_prefix.with_name(log_prefix.name + "_stdout.txt")
    stderr_path = log_prefix.with_name(log_prefix.name + "_stderr.txt")
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


def _parse_generation_positive_control(log_path: Path) -> dict[str, Any]:
    """从 VidSig generate_ms 日志读取视频级 bit accuracy。"""

    if not log_path.exists():
        raise FileNotFoundError(f"vidsig_generation_log_missing:{log_path}")
    bit_accuracy = None
    video_bit_accuracy = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "bit acc ms:" in line and ":" in line:
            bit_accuracy = float(line.rsplit(":", 1)[-1].strip())
        if "video bit acc ms:" in line and ":" in line:
            video_bit_accuracy = float(line.rsplit(":", 1)[-1].strip())
    if video_bit_accuracy is None:
        raise RuntimeError(f"vidsig_generation_video_bit_accuracy_missing:{log_path}")
    return {"official_generation_bit_accuracy": bit_accuracy, "official_generation_video_bit_accuracy": video_bit_accuracy}


def _parse_vidsig_attack_log(log_path: Path) -> tuple[float, bool]:
    """从 VidSig 官方 attack.py 日志中读取 `fpr = 1e-2` 的检测结果。"""

    if not log_path.exists():
        raise FileNotFoundError(f"vidsig_attack_log_missing:{log_path}")
    score = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "fpr = 1e-2" in line and ":" in line:
            score = float(line.rsplit(":", 1)[-1].strip())
    if score is None:
        raise RuntimeError(f"vidsig_detection_score_missing_in_log:{log_path}")
    return float(score), bool(score > 0.0)


def _read_video_frames(video_path: Path) -> list[Any]:
    """读取 mp4 帧序列。"""

    import imageio.v3 as iio

    return [frame for frame in iio.imiter(video_path)]


def _write_video_frames(video_path: Path, frames: list[Any], *, fps: int) -> None:
    """写出 mp4 帧序列。"""

    import imageio.v3 as iio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(video_path, frames, fps=fps)


def _apply_runtime_attack_to_frames(frames: list[Any], attack_name: str) -> tuple[list[Any], dict[str, Any]]:
    """对 VidSig watermarked 视频执行与主流程同名的 runtime attack。"""

    if not frames:
        raise ValueError("vidsig_no_decodable_frames")
    if attack_name == "video_compression_runtime":
        return list(frames), {"attack_transform": "decode_reencode", "attack_strength": "runtime_reencode_default_quality"}
    if attack_name == "temporal_crop_runtime":
        return (frames[1:-1] if len(frames) >= 4 else list(frames)), {
            "attack_transform": "drop_first_and_last_frame_when_possible",
            "attack_strength": "crop_boundary_frames",
        }
    if attack_name == "frame_rate_resampling_runtime":
        return (frames[::2] if len(frames) >= 3 else list(frames)), {
            "attack_transform": "keep_every_second_frame_when_possible",
            "attack_strength": "fps_downsample_by_2_proxy",
        }
    raise ValueError(f"unsupported_vidsig_runtime_attack:{attack_name}")


def _save_frame_array(frame_array_path: Path, frames: list[Any]) -> None:
    """把攻击后的帧序列保存成 VidSig attack.py 可读取的 `.npy`。"""

    import numpy as np

    frame_array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(frame_array_path, np.stack(frames, axis=0))


def _bundle_record_path(bundle_root: Path, record: Mapping[str, Any]) -> Path:
    """构造 VidSig official bundle 的单条记录路径。"""

    return (
        bundle_root
        / BASELINE_ID
        / "records"
        / f"{_safe_token(record.get('prompt_id'))}__{_safe_token(record.get('seed_id'))}__{_safe_token(record.get('attack_name'))}.json"
    )


def _unit_index_maps(prompt_rows: list[Mapping[str, Any]], seed_rows: list[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    """构造 VidSig 官方输出序号到 prompt / seed 的映射。"""

    mapping: dict[tuple[str, str], int] = {}
    seed_count = len(seed_rows)
    for prompt_index, prompt in enumerate(prompt_rows):
        for seed_index, seed in enumerate(seed_rows):
            mapping[(str(prompt["prompt_id"]), str(seed["seed_id"]))] = prompt_index * seed_count + seed_index
    return mapping


def _run_vidsig_attack_py(
    *,
    runtime_source_dir: Path,
    output_dir: Path,
    frame_array_dir: Path,
    config: VidSigOfficialRuntimeConfig,
) -> dict[str, Any]:
    """调用 VidSig 官方 attack.py 检测已经施加项目 runtime attack 的帧数组。"""

    command = [
        sys.executable,
        "src/attack.py",
        "--output_dir",
        str(output_dir),
        "--attack_type",
        "clean",
        "--factor",
        "2.0",
        "--frame_array_path",
        str(frame_array_dir),
        "--msg_decoder_path",
        config.msg_decoder_path,
        "--key",
        config.key,
    ]
    return _run_command(command, cwd=runtime_source_dir, log_prefix=output_dir / "official_attack_py", timeout_seconds=float(config.timeout_seconds))


def write_vidsig_official_bundle_records(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    manifest_path: str | Path,
    runtime_source_dir: str | Path,
    prompt_manifest: Mapping[str, Any],
    config: VidSigOfficialRuntimeConfig,
) -> dict[str, Any]:
    """把 VidSig 官方检测结果转写为 official bundle records。"""

    root = Path(run_root)
    bundle = Path(bundle_root)
    runtime_source = Path(runtime_source_dir)
    manifest = Path(manifest_path)
    records = _selected_runtime_records(root, config.max_records)
    unit_index_by_prompt_seed = _unit_index_maps(list(prompt_manifest.get("prompt_rows") or []), list(prompt_manifest.get("seed_rows") or []))
    clean_video_dir = Path(str(prompt_manifest["clean_video_dir"]))
    watermarked_video_dir = Path(str(prompt_manifest["watermarked_video_dir"]))
    generated = 0
    failures: list[dict[str, Any]] = []
    for record in records:
        output_json_path = _bundle_record_path(bundle, record)
        try:
            unit_index = unit_index_by_prompt_seed[(str(record.get("prompt_id") or ""), str(record.get("seed_id") or ""))]
            clean_video_path = clean_video_dir / f"{unit_index}.mp4"
            watermarked_video_path = watermarked_video_dir / f"{unit_index}.mp4"
            if not clean_video_path.exists():
                raise FileNotFoundError(f"vidsig_clean_video_missing:{clean_video_path}")
            if not watermarked_video_path.exists():
                raise FileNotFoundError(f"vidsig_watermarked_video_missing:{watermarked_video_path}")
            video_stem = output_json_path.stem
            official_record_work_dir = bundle / BASELINE_ID / "official_attack_work" / video_stem
            attacked_video_path = bundle / BASELINE_ID / "videos" / f"{video_stem}_attacked.mp4"
            clean_negative_video_path = bundle / BASELINE_ID / "videos" / f"{video_stem}_clean_negative.mp4"
            frame_array_dir = official_record_work_dir / "frame_arrays"
            frame_array_path = frame_array_dir / "sstw_attacked_video.npy"
            clean_frame_array_dir = official_record_work_dir / "clean_negative_frame_arrays"
            clean_frame_array_path = clean_frame_array_dir / "sstw_clean_negative_video.npy"
            frames = _read_video_frames(watermarked_video_path)
            clean_frames = _read_video_frames(clean_video_path)
            attacked_frames, attack_metadata = _apply_runtime_attack_to_frames(frames, str(record.get("attack_name") or ""))
            clean_negative_frames, clean_negative_attack_metadata = _apply_runtime_attack_to_frames(
                clean_frames,
                str(record.get("attack_name") or ""),
            )
            _write_video_frames(attacked_video_path, attacked_frames, fps=int(config.fps))
            _write_video_frames(clean_negative_video_path, clean_negative_frames, fps=int(config.fps))
            _save_frame_array(frame_array_path, attacked_frames)
            _save_frame_array(clean_frame_array_path, clean_negative_frames)
            attack_output_dir = official_record_work_dir / "official_attack_output"
            attack_result = _run_vidsig_attack_py(
                runtime_source_dir=runtime_source,
                output_dir=attack_output_dir,
                frame_array_dir=frame_array_dir,
                config=config,
            )
            if int(attack_result["return_code"]) != 0:
                raise RuntimeError(f"vidsig_official_attack_py_failed:{attack_result['return_code']}:{attack_result['stderr_tail']}")
            detection_score, detected = _parse_vidsig_attack_log(attack_output_dir / "log.txt")
            clean_attack_output_dir = official_record_work_dir / "official_clean_negative_attack_output"
            clean_attack_result = _run_vidsig_attack_py(
                runtime_source_dir=runtime_source,
                output_dir=clean_attack_output_dir,
                frame_array_dir=clean_frame_array_dir,
                config=config,
            )
            if int(clean_attack_result["return_code"]) != 0:
                raise RuntimeError(
                    "vidsig_official_clean_negative_attack_py_failed:"
                    f"{clean_attack_result['return_code']}:{clean_attack_result['stderr_tail']}"
                )
            clean_negative_score, _clean_detected = _parse_vidsig_attack_log(clean_attack_output_dir / "log.txt")
            payload = {
                "external_baseline_score": round(float(detection_score), 6),
                "raw_detector_score": round(float(detection_score), 6),
                "detection_score": round(float(detection_score), 6),
                "detected": bool(detected),
                "threshold": float(config.detection_threshold),
                "score_semantics": "official_tpr_at_fixed_fpr_detection_score",
                "score_orientation": "higher_is_more_watermarked",
                "external_baseline_clean_negative_score": round(float(clean_negative_score), 6),
                "external_baseline_clean_negative_score_semantics": "official_tpr_at_fixed_fpr_detection_score",
                "external_baseline_clean_negative_video_path": str(clean_negative_video_path),
                "official_vidsig_tpr_at_fpr_1e_2": round(float(detection_score), 6),
                "official_result_provenance": REPOSITORY_GENERATED_OFFICIAL_PROVENANCE,
                "official_adapter_baseline_id": BASELINE_ID,
                "official_baseline_id": BASELINE_ID,
                "external_baseline_generation_model_id": config.model_id,
                "external_baseline_official_execution_mode": "vidsig_generate_ms_watermarked_video_project_runtime_attack_official_attack_py",
                "official_score_extraction_policy": "vidsig_official_attack_log_tpr_at_fpr_1e_2_detection_score",
                "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                "attack_protocol_status": "project_runtime_attack_applied_to_vidsig_watermarked_video",
                "official_attack_type": "clean",
                "official_attack_factor": 2.0,
                "external_baseline_clean_video_path": str(clean_video_path),
                "external_baseline_source_video_path": str(watermarked_video_path),
                "external_baseline_attacked_video_path": str(attacked_video_path),
                "official_attacked_frame_array_path": str(frame_array_path),
                "official_clean_negative_frame_array_path": str(clean_frame_array_path),
                "official_attack_log_path": str(attack_output_dir / "log.txt"),
                "official_clean_negative_attack_log_path": str(clean_attack_output_dir / "log.txt"),
                "official_attack_stdout_path": attack_result["stdout_path"],
                "official_attack_stderr_path": attack_result["stderr_path"],
                "official_clean_negative_attack_stdout_path": clean_attack_result["stdout_path"],
                "official_clean_negative_attack_stderr_path": clean_attack_result["stderr_path"],
                "official_execution_manifest_path": str(manifest),
                "runtime_comparison_unit_id": build_comparison_unit_id(BASELINE_ID, record),
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "trajectory_trace_id": record.get("trajectory_trace_id"),
                "sstw_source_video_path": record.get("source_video_path"),
                "sstw_attacked_video_path": record.get("attacked_video_path"),
                "claim_support_status": "official_reference_bundle_written_not_claim_by_itself",
                **attack_metadata,
                "clean_negative_attack_transform": clean_negative_attack_metadata["attack_transform"],
                "clean_negative_attack_strength": clean_negative_attack_metadata["attack_strength"],
            }
            payload = {
                **payload,
                **official_score_formal_comparison_summary(payload),
                **official_score_formal_comparison_summary(payload, clean_negative=True),
            }
            _write_json(output_json_path, payload)
            generated += 1
        except Exception as exc:  # pragma: no cover - 单条视频 I/O 和官方 detector 依赖 Colab 环境
            failures.append({
                "baseline_id": BASELINE_ID,
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "failure_reason": str(exc),
            })
    return {"input_runtime_detection_record_count": len(records), "generated_bundle_record_count": generated, "failed_bundle_record_count": len(failures), "failures": failures[:20]}


def run_vidsig_official_runtime(config: VidSigOfficialRuntimeConfig) -> dict[str, Any]:
    """执行 VidSig 官方 generate_ms -> runtime attack -> attack.py 并生成 official bundle。"""

    run_root = Path(config.run_root)
    bundle_root = Path(config.bundle_root)
    source_dir = Path(config.source_dir)
    output_root = Path(config.output_root)
    prompt_suite_path = Path(config.prompt_suite_path)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    source_audit = _ensure_source_ready(source_dir)
    resource_audit = [
        _ensure_resource_file(config.msg_decoder_path, env_name="SSTW_VIDSIG_MSG_DECODER_PATH", role="official_message_decoder_checkpoint"),
        _ensure_resource_file(config.vae_checkpoint_path, env_name="SSTW_VIDSIG_VAE_CHECKPOINT_PATH", role="official_vae_checkpoint_for_generate_ms"),
    ]
    records = _selected_runtime_records(run_root, config.max_records)
    if not records:
        raise RuntimeError(f"vidsig_runtime_detection_records_missing:{run_root / 'records/runtime_detection_records.jsonl'}")
    prompt_rows, seed_rows = _runtime_prompt_seed_rows(
        records,
        prompt_suite_path=prompt_suite_path,
        allow_prompt_id_fallback=config.allow_prompt_id_fallback,
        allow_seed_id_fallback=config.allow_seed_id_fallback,
    )
    runtime_source_dir = output_root / "source_runtime"
    _copy_official_source_to_runtime(source_dir, runtime_source_dir, output_root, force=config.force_rebuild_runtime_source)
    runtime_patch_audit = _patch_generate_ms_for_colab(runtime_source_dir)
    prompt_manifest = _write_vidsig_generation_inputs(
        runtime_source_dir=runtime_source_dir,
        output_root=output_root,
        config=config,
        prompt_rows=prompt_rows,
        seed_rows=seed_rows,
    )
    generate_command = [sys.executable, "src/generate_ms.py"]
    command_results: list[dict[str, Any]] = []
    execution_failure_reason = ""
    execution_status = "dry_run_planned" if config.dry_run else "executed"
    positive_control_result: dict[str, Any] = {"positive_control_status": "dry_run_not_checked" if config.dry_run else "pending"}
    if not config.dry_run:
        generate_result = _run_command(
            generate_command,
            cwd=runtime_source_dir,
            log_prefix=output_root / "logs" / "vidsig_generate_ms",
            timeout_seconds=float(config.timeout_seconds),
        )
        command_results.append(generate_result)
        if int(generate_result["return_code"]) != 0:
            execution_failure_reason = f"vidsig_generate_ms_failed:{generate_result['return_code']}:{generate_result['stderr_tail']}"
        if not execution_failure_reason:
            try:
                positive_control_result = _parse_generation_positive_control(Path(str(prompt_manifest["generation_log_path"])))
                positive_score = float(positive_control_result["official_generation_video_bit_accuracy"])
                positive_control_result["positive_control_threshold"] = float(config.positive_control_bit_accuracy_threshold)
                positive_control_result["positive_control_status"] = "PASS" if positive_score >= float(config.positive_control_bit_accuracy_threshold) else "FAIL"
                if positive_control_result["positive_control_status"] != "PASS":
                    execution_failure_reason = (
                        "vidsig_generation_positive_control_failed:"
                        f"video_bit_accuracy={positive_score}:threshold={config.positive_control_bit_accuracy_threshold}"
                    )
            except Exception as exc:
                positive_control_result = {
                    "positive_control_status": "FAIL",
                    "failure_reason": str(exc),
                    "positive_control_threshold": float(config.positive_control_bit_accuracy_threshold),
                }
                execution_failure_reason = str(exc)

    manifest_path = bundle_root / BASELINE_ID / "official_reference_execution_manifest.json"
    if not config.dry_run and not execution_failure_reason:
        bundle_result = write_vidsig_official_bundle_records(
            run_root=run_root,
            bundle_root=bundle_root,
            manifest_path=manifest_path,
            runtime_source_dir=runtime_source_dir,
            prompt_manifest=prompt_manifest,
            config=config,
        )
    elif execution_failure_reason:
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
    else:
        bundle_result = {"input_runtime_detection_record_count": len(records), "generated_bundle_record_count": 0, "failed_bundle_record_count": 0, "failures": []}

    manifest = {
        "manifest_kind": "modern_external_baseline_formal_reference_execution_manifest",
        "baseline_id": BASELINE_ID,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "official_source_dir": str(source_dir),
        "runtime_source_dir": str(runtime_source_dir),
        "official_runtime_output_root": str(output_root),
        "official_repository_url": "https://github.com/hardenyu21/Video-Signature",
        "official_execution_mode": "vidsig_generate_ms_watermarked_video_project_runtime_attack_official_attack_py",
        "execution_status": execution_status,
        "execution_failure_reason": execution_failure_reason,
        "dry_run": bool(config.dry_run),
        "config": asdict(config),
        "source_audit": source_audit,
        "resource_audit": resource_audit,
        "runtime_patch_audit": runtime_patch_audit,
        "prompt_manifest": prompt_manifest,
        "generate_command": generate_command,
        "command_results": command_results,
        "positive_control_result": positive_control_result,
        "input_runtime_detection_record_count": len(records),
        "generated_video_unit_count": len(prompt_rows) * len(seed_rows),
        "generated_bundle_record_count": int(bundle_result["generated_bundle_record_count"]),
        "failed_bundle_record_count": int(bundle_result["failed_bundle_record_count"]),
        "failures": bundle_result.get("failures", []),
        "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
    }
    _write_json(manifest_path, manifest)
    _write_json(output_root / "vidsig_official_runtime_execution_manifest.json", manifest)
    os.environ["SSTW_VIDSIG_GENERATED_OFFICIAL_OUTPUT_ROOT"] = str(prompt_manifest["official_output_root"])
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行入口参数。"""

    parser = argparse.ArgumentParser(description="运行 VidSig 官方 generate_ms->attack.py 并生成 official bundle")
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
    config = build_default_vidsig_official_config_from_env(
        run_root=args.run_root,
        bundle_root=args.bundle_root,
        source_dir=args.source_dir,
        repo_root=args.repo_root,
        resource_root=args.resource_root or None,
        max_records=args.max_records,
    )
    if args.dry_run:
        config = VidSigOfficialRuntimeConfig(**{**asdict(config), "dry_run": True})
    result = run_vidsig_official_runtime(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI 入口由 Notebook 调用
    main()
