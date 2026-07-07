"""WAM-frame 官方流程的项目内 official bundle 生成器。

WAM 本体是图像水印方法。该模块只把它作为 frame-wise image watermark adapted
to video 的 baseline: 对同一 clean video 的帧逐帧嵌入 WAM 水印, 施加项目
runtime attack, 再逐帧检测并用视频级 bit accuracy 均值作为分数。clean negative
使用同一 baseline 的无水印帧经相同 attack 后检测得到, 以支持 target FPR 校准。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import urllib.error
import urllib.request

from external_baseline.official_eval_adapters.common import build_official_reference_bundle_execution_status
from external_baseline.official_runtime_progress import emit_official_reference_plan
from external_baseline.runtime_trace_io import build_comparison_unit_id, comparable_detection_records
from external_baseline.score_semantics import official_score_formal_comparison_summary
from external_baseline.video_tensor_io import read_video_tchw_uint8, write_video_tchw
from main.attacks.video_runtime_attack_protocol import apply_runtime_attack_to_video_tensor
from main.core.digest import build_stable_digest
from main.core.progress import ProgressReporter, suppress_third_party_progress_output


BASELINE_ID = "wam_frame"
REPOSITORY_PROVENANCE = "repository_generated_from_third_party_official_code"
WAM_MIT_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth"


@dataclass(frozen=True)
class WAMFrameOfficialRuntimeConfig:
    """WAM-frame official bundle 生成所需的最小配置。"""

    run_root: str
    bundle_root: str
    source_dir: str
    repo_root: str
    resource_root: str
    max_records: int | None = None
    device: str | None = None
    max_frames: int = 8
    message_bits: int = 32

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写出稳定 JSON artifact。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_token(value: Any) -> str:
    """把 prompt、seed 和 attack 转换为文件名 token。"""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return text.strip("_") or "unknown"


def _bundle_record_path(bundle_root: Path, record: Mapping[str, Any]) -> Path:
    """构造单条 official bundle JSON 路径。"""

    return (
        bundle_root
        / BASELINE_ID
        / "records"
        / f"{_safe_token(record.get('prompt_id'))}__{_safe_token(record.get('seed_id'))}__{_safe_token(record.get('attack_name'))}.json"
    )


def _deterministic_bits(record: Mapping[str, Any], bit_count: int) -> list[int]:
    """基于 comparison unit 生成可复现 WAM message bits。"""

    seed_payload = {
        "baseline_id": BASELINE_ID,
        "prompt_id": record.get("prompt_id"),
        "seed_id": record.get("seed_id"),
        "trajectory_trace_id": record.get("trajectory_trace_id"),
        "bit_count": int(bit_count),
    }
    digest = build_stable_digest(seed_payload)
    bits: list[int] = []
    counter = 0
    while len(bits) < bit_count:
        block = build_stable_digest({"digest": digest, "counter": counter})
        for char in block:
            value = int(char, 16)
            bits.extend([(value >> shift) & 1 for shift in range(3, -1, -1)])
            if len(bits) >= bit_count:
                break
        counter += 1
    return bits[:bit_count]


def _path_candidates(raw_path: str, *, repo_root: Path, source_dir: Path) -> list[Path]:
    """把官方资源路径解析为可检查候选路径。

    Colab resource bootstrap 会把部分路径写成仓库相对路径。官方 WAM 代码又要求
    在 source 目录作为 cwd 时读取相对配置文件。因此 runtime 层必须先把环境变量中的
    路径解析为绝对路径, 再临时切换 cwd 调用官方 loader。
    """

    raw = Path(raw_path).expanduser()
    if raw.is_absolute():
        return [raw]
    return [repo_root / raw, source_dir / raw, raw]


def _resolve_existing_file(
    raw_path: str,
    *,
    repo_root: Path,
    source_dir: Path,
    fallback: Path,
    missing_label: str,
) -> Path:
    """解析必须存在的文件路径, 并在失败时报告所有候选路径。"""

    candidates = _path_candidates(raw_path, repo_root=repo_root, source_dir=source_dir) if raw_path else [fallback]
    if fallback not in candidates:
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{missing_label}:{[str(candidate) for candidate in candidates]}")


def _resolve_checkpoint(source_dir: Path, resource_root: Path, repo_root: Path) -> Path:
    """解析 WAM checkpoint, 必要时从官方公开 URL 下载 MIT 权重。"""

    env_path = os.environ.get("SSTW_WAM_FRAME_CHECKPOINT_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.extend(_path_candidates(env_path, repo_root=repo_root, source_dir=source_dir))
    candidates.extend([
        resource_root / "wam_frame" / "wam_mit.pth",
        resource_root / "wam_frame" / "checkpoint.pth",
        source_dir / "checkpoints" / "wam_mit.pth",
        source_dir / "checkpoints" / "checkpoint.pth",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    target = resource_root / "wam_frame" / "wam_mit.pth"
    if os.environ.get("SSTW_WAM_FRAME_DISABLE_CHECKPOINT_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}:
        raise FileNotFoundError(f"wam_frame_checkpoint_missing:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(WAM_MIT_CHECKPOINT_URL, target)
    except (OSError, urllib.error.URLError) as exc:
        raise FileNotFoundError(f"wam_frame_checkpoint_download_failed:{target}:{exc}") from exc
    return target.resolve()


def _load_wam_model(config: WAMFrameOfficialRuntimeConfig) -> tuple[Any, Any, Any, Any, str, Path]:
    """加载 WAM 官方模型和 transform 函数。"""

    import torch

    source_dir = Path(config.source_dir).resolve()
    repo_root = Path(config.repo_root).resolve() if config.repo_root else Path.cwd().resolve()
    resource_root = Path(config.resource_root).resolve() if config.resource_root else source_dir / "checkpoints"
    sys.path.insert(0, str(source_dir))
    from notebooks.inference_utils import load_model_from_checkpoint  # type: ignore
    from watermark_anything.data.metrics import msg_predict_inference  # type: ignore
    from watermark_anything.data.transforms import default_transform, unnormalize_img  # type: ignore

    params_path = _resolve_existing_file(
        os.environ.get("SSTW_WAM_FRAME_PARAMS_PATH", "").strip(),
        repo_root=repo_root,
        source_dir=source_dir,
        fallback=source_dir / "checkpoints" / "params.json",
        missing_label="wam_frame_params_missing",
    )
    checkpoint_path = _resolve_checkpoint(source_dir, resource_root, repo_root)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cwd = Path.cwd()
    try:
        os.chdir(source_dir)
        model = load_model_from_checkpoint(str(params_path), str(checkpoint_path)).to(device).eval()
    finally:
        os.chdir(cwd)
    return model, default_transform, unnormalize_img, msg_predict_inference, device, checkpoint_path


def _pil_from_chw_uint8(frame: Any) -> Any:
    """把 `[C, H, W]` uint8 frame 转为 PIL RGB image。"""

    from PIL import Image

    array = frame.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(array.astype("uint8"), mode="RGB")


def _select_frames(video: Any, max_frames: int) -> Any:
    """选择 WAM-frame 运行帧, 保持正负样本使用同一策略。"""

    if max_frames <= 0 or int(video.shape[0]) <= max_frames:
        return video
    return video[:max_frames]


def _frames_to_wam_tensor(video: Any, transform: Any, *, device: str, max_frames: int) -> Any:
    """把视频帧转换成 WAM 官方 transform 后的 batch tensor。"""

    import torch

    selected = _select_frames(video, max_frames)
    frames = [transform(_pil_from_chw_uint8(frame)) for frame in selected]
    if not frames:
        raise RuntimeError("wam_frame_video_empty_after_frame_selection")
    return torch.stack(frames, dim=0).to(device)


def _detect_wam_score(model: Any, video: Any, transform: Any, msg_predict_inference: Any, message: Any, *, device: str, max_frames: int) -> tuple[float, float]:
    """用 WAM 官方 detect 输出计算视频级 bit accuracy 和 mask confidence。"""

    import torch
    import torch.nn.functional as F

    batch = _frames_to_wam_tensor(video, transform, device=device, max_frames=max_frames)
    with torch.no_grad():
        preds = model.detect(batch)["preds"]
    mask_preds = F.sigmoid(preds[:, 0:1, :, :])
    bit_preds = preds[:, 1:, :, :]
    pred_message = msg_predict_inference(bit_preds, mask_preds).to(device).float()
    reference = message.reshape(1, -1).repeat(pred_message.shape[0], 1)
    bit_accuracy = float((pred_message == reference).float().mean().item())
    mask_confidence = float(mask_preds.mean().item())
    return bit_accuracy, mask_confidence


def _embed_wam_video(model: Any, video: Any, transform: Any, unnormalize_img: Any, message: Any, *, device: str, max_frames: int) -> Any:
    """对视频帧逐帧执行 WAM 官方 embed。"""

    import torch

    batch = _frames_to_wam_tensor(video, transform, device=device, max_frames=max_frames)
    watermarked_frames = []
    with torch.no_grad():
        for frame in batch:
            outputs = model.embed(frame.unsqueeze(0), message)
            watermarked_frames.append(unnormalize_img(outputs["imgs_w"]).clamp(0.0, 1.0).squeeze(0).detach().cpu())
    if not watermarked_frames:
        raise RuntimeError("wam_frame_embed_no_frames")
    return torch.stack(watermarked_frames, dim=0).contiguous()


def _payload_with_formal_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """补齐 official score 粒度和 clean negative 粒度摘要。"""

    merged = dict(payload)
    merged.update(official_score_formal_comparison_summary(merged))
    merged.update(official_score_formal_comparison_summary(merged, clean_negative=True))
    return merged


def build_default_wam_frame_official_config_from_env(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    source_dir: str | Path,
    repo_root: str | Path = ".",
    resource_root: str | Path = "",
    max_records: int | None = None,
) -> WAMFrameOfficialRuntimeConfig:
    """从环境变量构造 WAM-frame 默认运行配置。"""

    return WAMFrameOfficialRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        repo_root=str(repo_root),
        resource_root=str(resource_root),
        max_records=max_records,
        device=os.environ.get("SSTW_WAM_FRAME_DEVICE", "").strip() or None,
        max_frames=int(os.environ.get("SSTW_WAM_FRAME_MAX_FRAMES", "8")),
        message_bits=int(os.environ.get("SSTW_WAM_FRAME_MESSAGE_BITS", "32")),
    )


def run_wam_frame_official_runtime(config: WAMFrameOfficialRuntimeConfig) -> dict[str, Any]:
    """运行 WAM 官方 frame-wise embed / detect 并生成 official bundle。"""

    import torch

    run_root = Path(config.run_root)
    bundle_root = Path(config.bundle_root)
    source_dir = Path(config.source_dir)
    resource_root = Path(config.resource_root) if config.resource_root else source_dir / "checkpoints"
    records = comparable_detection_records(run_root)
    if config.max_records is not None:
        records = records[: int(config.max_records)]
    emit_official_reference_plan(
        BASELINE_ID,
        runtime_detection_record_count=len(records),
        runtime_attack_count=len({str(record.get("attack_name")) for record in records if record.get("attack_name")}),
        extra="official_steps=frame_embed,attack,frame_detect,clean_negative_detect",
    )
    model, transform, unnormalize_img, msg_predict_inference, device, checkpoint_path = _load_wam_model(config)
    generated = 0
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    baseline_video_dir = bundle_root / BASELINE_ID / "videos"
    manifest_path = bundle_root / BASELINE_ID / "official_reference_execution_manifest.json"
    progress = ProgressReporter("official_bundle_generation:wam_frame", len(records), "runtime_video")

    for index, record in enumerate(records, start=1):
        progress.update(index, f"prompt={record.get('prompt_id')} seed={record.get('seed_id')} attack={record.get('attack_name')}")
        output_json = _bundle_record_path(bundle_root, record)
        try:
            source_video_path = Path(str(record.get("source_video_path") or ""))
            if not source_video_path.exists():
                raise FileNotFoundError(f"source_video_missing:{source_video_path}")
            source_video, source_info = read_video_tchw_uint8(source_video_path, empty_error="wam_frame_source_video_empty")
            fps = float(source_info.get("video_fps") or 8.0)
            reference_bits = _deterministic_bits(record, config.message_bits)
            message = torch.tensor(reference_bits, dtype=torch.float32, device=device).reshape(1, -1)
            with suppress_third_party_progress_output("official_reference_embed:wam_frame"):
                watermarked = _embed_wam_video(
                    model,
                    source_video,
                    transform,
                    unnormalize_img,
                    message,
                    device=device,
                    max_frames=config.max_frames,
                )
            clean_base = _select_frames(source_video, config.max_frames).float() / 255.0
            attacked = apply_runtime_attack_to_video_tensor(watermarked, str(record.get("attack_name") or ""))
            clean_negative = apply_runtime_attack_to_video_tensor(clean_base, str(record.get("attack_name") or ""))
            video_stem = output_json.stem
            watermarked_path = baseline_video_dir / f"{video_stem}_watermarked.mp4"
            attacked_path = baseline_video_dir / f"{video_stem}_attacked.mp4"
            clean_negative_path = baseline_video_dir / f"{video_stem}_clean_negative.mp4"
            write_video_tchw(watermarked_path, watermarked, fps=fps)
            write_video_tchw(attacked_path, attacked, fps=fps)
            write_video_tchw(clean_negative_path, clean_negative, fps=fps)
            attacked_read, attacked_info = read_video_tchw_uint8(attacked_path, empty_error="wam_frame_attacked_video_empty_after_reencode")
            clean_negative_read, clean_info = read_video_tchw_uint8(clean_negative_path, empty_error="wam_frame_clean_negative_video_empty_after_reencode")
            with suppress_third_party_progress_output("official_reference_detect:wam_frame"):
                score, confidence = _detect_wam_score(
                    model,
                    attacked_read,
                    transform,
                    msg_predict_inference,
                    message,
                    device=device,
                    max_frames=config.max_frames,
                )
                clean_score, clean_confidence = _detect_wam_score(
                    model,
                    clean_negative_read,
                    transform,
                    msg_predict_inference,
                    message,
                    device=device,
                    max_frames=config.max_frames,
                )
            payload = _payload_with_formal_summary({
                "external_baseline_score": round(float(score), 6),
                "raw_detector_score": round(float(score), 6),
                "bit_accuracy": round(float(score), 6),
                "payload_bit_accuracy": round(float(score), 6),
                "confidence": round(float(confidence), 6),
                "detected": bool(score >= float(os.environ.get("SSTW_WAM_FRAME_DETECTION_THRESHOLD", "0.75"))),
                "threshold": float(os.environ.get("SSTW_WAM_FRAME_DETECTION_THRESHOLD", "0.75")),
                "score_semantics": "payload_bit_accuracy_extraction_score",
                "score_orientation": "higher_is_more_watermarked",
                "official_score_extraction_policy": "wam_frame_official_frame_detect_bit_accuracy_mean_per_prompt_seed_attack",
                "official_score_granularity": "per_prompt_seed_attack",
                "official_score_value_type": "payload_bit_accuracy_score",
                "official_clean_negative_score_granularity": "per_prompt_seed_attack",
                "official_clean_negative_score_value_type": "payload_bit_accuracy_score",
                "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                "attack_protocol_status": "wam_frame_official_frame_embed_then_project_runtime_attack",
                "external_baseline_clean_negative_score": round(float(clean_score), 6),
                "external_baseline_clean_negative_score_semantics": "payload_bit_accuracy_extraction_score",
                "external_baseline_clean_negative_video_path": str(clean_negative_path),
                "external_baseline_clean_negative_confidence": round(float(clean_confidence), 6),
                "external_baseline_source_video_path": str(watermarked_path),
                "external_baseline_attacked_video_path": str(attacked_path),
                "external_baseline_generation_model_id": "wam_frame_official_image_watermark_adapter",
                "external_baseline_official_execution_mode": "wam_official_framewise_embed_detect",
                "official_result_provenance": REPOSITORY_PROVENANCE,
                "official_adapter_baseline_id": BASELINE_ID,
                "official_baseline_id": BASELINE_ID,
                "official_result_bundle_path": str(output_json),
                "official_execution_manifest_path": str(manifest_path),
                "official_source_dir": str(source_dir),
                "official_checkpoint_path": str(checkpoint_path),
                "official_video_frame_count": int(min(int(source_video.shape[0]), config.max_frames) if config.max_frames > 0 else int(source_video.shape[0])),
                "official_frame_adapter_policy": "frame_wise_image_watermark_with_video_level_mean_bit_accuracy",
                "official_payload_message_digest": build_stable_digest(reference_bits),
                "official_video_io_backend": source_info.get("video_io_backend"),
                "official_attacked_video_io_backend": attacked_info.get("video_io_backend"),
                "official_clean_negative_video_io_backend": clean_info.get("video_io_backend"),
                "runtime_comparison_unit_id": build_comparison_unit_id(BASELINE_ID, record),
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "trajectory_trace_id": record.get("trajectory_trace_id"),
                "source_sstw_video_path": str(source_video_path),
                "sstw_attacked_video_path": str(record.get("attacked_video_path") or ""),
            })
            _write_json(output_json, payload)
            generated += 1
            successes.append({
                "official_output_json_path": str(output_json),
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
            })
        except Exception as exc:  # pragma: no cover - 依赖第三方官方代码、视频编解码和 GPU/CPU 张量运行。
            failures.append({
                "baseline_id": BASELINE_ID,
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "failure_reason": str(exc),
            })
    progress.finish(f"generated={generated} failed={len(failures)}")
    manifest = {
        "manifest_kind": "modern_external_baseline_formal_reference_execution_manifest",
        "baseline_id": BASELINE_ID,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "official_source_dir": str(source_dir),
        "resource_root": str(resource_root),
        "execution_status": build_official_reference_bundle_execution_status(
            generated_count=generated,
            expected_count=len(records),
            failed_count=len(failures),
        ),
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": generated,
        "failed_bundle_record_count": len(failures),
        "successes": successes[:20],
        "failures": failures[:20],
        "config": config.to_dict(),
        "claim_support_status": "official_reference_bundle_ready_not_measured_formal_record" if generated == len(records) and records and not failures else "official_reference_bundle_blocked_not_claim_evidence",
    }
    _write_json(manifest_path, manifest)
    return manifest
