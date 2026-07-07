"""REVMark 官方流程的项目内 official bundle 生成器。

该模块把 REVMark 的官方 Encoder / Decoder 代码接入 SSTW 的公平比较协议。
它以当前 runtime detection records 中的 prompt / seed / attack 为锚点,
对同一 clean video 执行 REVMark 嵌入, 再施加项目 runtime attack, 最后用
REVMark 官方 Decoder 提取消息并写出 per-sample official bundle。clean negative
使用同一 baseline 的 clean video 经相同 attack 后检测得到, 用于后续 target FPR
校准。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from external_baseline.official_eval_adapters.common import build_official_reference_bundle_execution_status
from external_baseline.official_runtime_progress import emit_official_reference_plan
from external_baseline.runtime_trace_io import build_comparison_unit_id, comparable_detection_records
from external_baseline.score_semantics import official_score_formal_comparison_summary
from external_baseline.video_tensor_io import read_video_tchw_uint8, write_video_tchw
from main.attacks.video_runtime_attack_protocol import apply_runtime_attack_to_video_tensor
from main.core.digest import build_stable_digest
from main.core.progress import ProgressReporter, suppress_third_party_progress_output


BASELINE_ID = "revmark"
REPOSITORY_PROVENANCE = "repository_generated_from_third_party_official_code"


@dataclass(frozen=True)
class REVMarkOfficialRuntimeConfig:
    """REVMark official bundle 生成所需的最小配置。"""

    run_root: str
    bundle_root: str
    source_dir: str
    repo_root: str
    resource_root: str
    max_records: int | None = None
    device: str | None = None
    frame_count: int = 8
    frame_size: int = 128
    message_bits: int = 96
    embed_strength: float = 6.2

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
    """基于 comparison unit 生成可复现消息位。"""

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


def _prepare_revmark_video(video: Any, *, frame_count: int, frame_size: int, device: str) -> Any:
    """把任意视频转换为 REVMark 官方模型需要的 `[1, C, T, H, W]` 张量。"""

    import torch
    import torch.nn.functional as F

    tensor = video.float() / 255.0
    if tensor.shape[0] < frame_count:
        repeat_count = frame_count - int(tensor.shape[0])
        tensor = torch.cat([tensor, tensor[-1:].repeat(repeat_count, 1, 1, 1)], dim=0)
    tensor = tensor[:frame_count]
    tensor = F.interpolate(tensor, size=(frame_size, frame_size), mode="bilinear", align_corners=False)
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return (tensor.to(device) * 2.0) - 1.0


def _model_video_to_tchw_unit_range(video_bcthw: Any) -> Any:
    """把 REVMark `[1, C, T, H, W]` 输出转回 `[T, C, H, W]` 的 0 到 1 张量。"""

    return ((video_bcthw.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0).squeeze(0).permute(1, 0, 2, 3).contiguous()


def _bit_accuracy(predicted: Any, reference_bits: list[int]) -> float:
    """计算 REVMark decoded message 与参考消息的 bit accuracy。"""

    import torch

    reference = torch.tensor(reference_bits, dtype=torch.float32, device=predicted.device).reshape_as(predicted)
    return float(((predicted >= 0.5) == (reference >= 0.5)).float().mean().item())


def _load_revmark_models(config: REVMarkOfficialRuntimeConfig) -> tuple[Any, Any, Any, str]:
    """加载 REVMark 官方 Encoder / Decoder 与 framenorm 函数。"""

    import torch

    source_dir = Path(config.source_dir).resolve()
    sys.path.insert(0, str(source_dir))
    from REVMark import Decoder, Encoder, framenorm  # type: ignore

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    motion_estimation_checkpoint = source_dir / "ME_Spynet_Full.pth"
    if not motion_estimation_checkpoint.exists():
        raise FileNotFoundError(f"revmark_motion_estimation_checkpoint_missing:{motion_estimation_checkpoint}")
    encoder_checkpoint = Path(os.environ.get("SSTW_REVMARK_ENCODER_CHECKPOINT_PATH", "").strip() or source_dir / "checkpoints" / "Encoder.pth")
    decoder_checkpoint = Path(os.environ.get("SSTW_REVMARK_DECODER_CHECKPOINT_PATH", "").strip() or source_dir / "checkpoints" / "Decoder.pth")
    if not encoder_checkpoint.exists():
        raise FileNotFoundError(f"revmark_encoder_checkpoint_missing:{encoder_checkpoint}")
    if not decoder_checkpoint.exists():
        raise FileNotFoundError(f"revmark_decoder_checkpoint_missing:{decoder_checkpoint}")
    encoder_checkpoint = encoder_checkpoint.resolve()
    decoder_checkpoint = decoder_checkpoint.resolve()
    cwd = Path.cwd()
    try:
        # REVMark 官方 TAsBlock 在构造时以相对路径读取 ME_Spynet_Full.pth。
        # Colab notebook 的 cwd 是仓库根目录, 因此这里临时切到官方 source
        # 目录, 既保持第三方源码不被修改, 又保证官方相对路径语义成立。
        os.chdir(source_dir)
        encoder = Encoder(config.message_bits, [config.frame_count, config.frame_size, config.frame_size]).to(device).eval()
        decoder = Decoder(config.message_bits, [config.frame_count, config.frame_size, config.frame_size]).to(device).eval()
        encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=device))
        decoder.load_state_dict(torch.load(decoder_checkpoint, map_location=device))
    finally:
        os.chdir(cwd)
    if hasattr(encoder, "tasblock"):
        encoder.tasblock.enable = True
    if hasattr(decoder, "tasblock"):
        decoder.tasblock.enable = True
    return encoder, decoder, framenorm, device


def _decode_score(decoder: Any, video_tchw: Any, reference_bits: list[int], *, config: REVMarkOfficialRuntimeConfig, device: str) -> float:
    """使用 REVMark 官方 Decoder 对视频解码并返回 bit accuracy 分数。"""

    import torch

    prepared = _prepare_revmark_video(video_tchw, frame_count=config.frame_count, frame_size=config.frame_size, device=device)
    with torch.no_grad():
        decoded = decoder(prepared)
    return _bit_accuracy(decoded, reference_bits)


def _payload_with_formal_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """补齐 official score 粒度和 clean negative 粒度摘要。"""

    merged = dict(payload)
    merged.update(official_score_formal_comparison_summary(merged))
    merged.update(official_score_formal_comparison_summary(merged, clean_negative=True))
    return merged


def build_default_revmark_official_config_from_env(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    source_dir: str | Path,
    repo_root: str | Path = ".",
    resource_root: str | Path = "",
    max_records: int | None = None,
) -> REVMarkOfficialRuntimeConfig:
    """从环境变量构造 REVMark 默认运行配置。"""

    return REVMarkOfficialRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        repo_root=str(repo_root),
        resource_root=str(resource_root),
        max_records=max_records,
        device=os.environ.get("SSTW_REVMARK_DEVICE", "").strip() or None,
        frame_count=int(os.environ.get("SSTW_REVMARK_FRAME_COUNT", "8")),
        frame_size=int(os.environ.get("SSTW_REVMARK_FRAME_SIZE", "128")),
        message_bits=int(os.environ.get("SSTW_REVMARK_MESSAGE_BITS", "96")),
        embed_strength=float(os.environ.get("SSTW_REVMARK_EMBED_STRENGTH", "6.2")),
    )


def run_revmark_official_runtime(config: REVMarkOfficialRuntimeConfig) -> dict[str, Any]:
    """运行 REVMark 官方 Encoder / Decoder 并生成 official bundle。"""

    import torch

    run_root = Path(config.run_root)
    bundle_root = Path(config.bundle_root)
    source_dir = Path(config.source_dir)
    records = comparable_detection_records(run_root)
    if config.max_records is not None:
        records = records[: int(config.max_records)]
    emit_official_reference_plan(
        BASELINE_ID,
        runtime_detection_record_count=len(records),
        runtime_attack_count=len({str(record.get("attack_name")) for record in records if record.get("attack_name")}),
        extra="official_steps=encode,attack,decode,clean_negative_decode",
    )
    encoder, decoder, framenorm, device = _load_revmark_models(config)
    generated = 0
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    baseline_video_dir = bundle_root / BASELINE_ID / "videos"
    manifest_path = bundle_root / BASELINE_ID / "official_reference_execution_manifest.json"
    progress = ProgressReporter("official_bundle_generation:revmark", len(records), "runtime_video")

    for index, record in enumerate(records, start=1):
        progress.update(index, f"prompt={record.get('prompt_id')} seed={record.get('seed_id')} attack={record.get('attack_name')}")
        output_json = _bundle_record_path(bundle_root, record)
        try:
            source_video_path = Path(str(record.get("source_video_path") or ""))
            if not source_video_path.exists():
                raise FileNotFoundError(f"source_video_missing:{source_video_path}")
            source_video, source_info = read_video_tchw_uint8(source_video_path, empty_error="revmark_source_video_empty")
            fps = float(source_info.get("video_fps") or 8.0)
            reference_bits = _deterministic_bits(record, config.message_bits)
            message = torch.tensor(reference_bits, dtype=torch.float32, device=device).unsqueeze(0)
            cover = _prepare_revmark_video(
                source_video,
                frame_count=config.frame_count,
                frame_size=config.frame_size,
                device=device,
            )
            with suppress_third_party_progress_output("official_reference_embed:revmark"):
                with torch.no_grad():
                    residual = encoder(cover, message)
                    stego = (cover + config.embed_strength * framenorm(residual)).clamp(-1.0, 1.0)
            watermarked = _model_video_to_tchw_unit_range(stego)
            attacked = apply_runtime_attack_to_video_tensor(watermarked, str(record.get("attack_name") or ""))
            clean_negative_base = _model_video_to_tchw_unit_range(cover)
            clean_negative = apply_runtime_attack_to_video_tensor(clean_negative_base, str(record.get("attack_name") or ""))
            video_stem = output_json.stem
            watermarked_path = baseline_video_dir / f"{video_stem}_watermarked.mp4"
            attacked_path = baseline_video_dir / f"{video_stem}_attacked.mp4"
            clean_negative_path = baseline_video_dir / f"{video_stem}_clean_negative.mp4"
            write_video_tchw(watermarked_path, watermarked, fps=fps)
            write_video_tchw(attacked_path, attacked, fps=fps)
            write_video_tchw(clean_negative_path, clean_negative, fps=fps)
            attacked_read, attacked_info = read_video_tchw_uint8(attacked_path, empty_error="revmark_attacked_video_empty_after_reencode")
            clean_negative_read, clean_info = read_video_tchw_uint8(clean_negative_path, empty_error="revmark_clean_negative_video_empty_after_reencode")
            score = _decode_score(decoder, attacked_read, reference_bits, config=config, device=device)
            clean_score = _decode_score(decoder, clean_negative_read, reference_bits, config=config, device=device)
            payload = _payload_with_formal_summary({
                "external_baseline_score": round(float(score), 6),
                "raw_detector_score": round(float(score), 6),
                "bit_accuracy": round(float(score), 6),
                "payload_bit_accuracy": round(float(score), 6),
                "detected": bool(score >= float(os.environ.get("SSTW_REVMARK_DETECTION_THRESHOLD", "0.75"))),
                "threshold": float(os.environ.get("SSTW_REVMARK_DETECTION_THRESHOLD", "0.75")),
                "score_semantics": "payload_bit_accuracy_extraction_score",
                "score_orientation": "higher_is_more_watermarked",
                "official_score_extraction_policy": "revmark_official_decoder_bit_accuracy_per_prompt_seed_attack",
                "official_score_granularity": "per_prompt_seed_attack",
                "official_score_value_type": "payload_bit_accuracy_score",
                "official_clean_negative_score_granularity": "per_prompt_seed_attack",
                "official_clean_negative_score_value_type": "payload_bit_accuracy_score",
                "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                "attack_protocol_status": "revmark_official_embed_then_project_runtime_attack",
                "external_baseline_clean_negative_score": round(float(clean_score), 6),
                "external_baseline_clean_negative_score_semantics": "payload_bit_accuracy_extraction_score",
                "external_baseline_clean_negative_video_path": str(clean_negative_path),
                "external_baseline_source_video_path": str(watermarked_path),
                "external_baseline_attacked_video_path": str(attacked_path),
                "external_baseline_generation_model_id": "revmark_official_encoder_decoder",
                "external_baseline_official_execution_mode": "revmark_official_encoder_decoder",
                "official_result_provenance": REPOSITORY_PROVENANCE,
                "official_adapter_baseline_id": BASELINE_ID,
                "official_baseline_id": BASELINE_ID,
                "official_result_bundle_path": str(output_json),
                "official_execution_manifest_path": str(manifest_path),
                "official_source_dir": str(source_dir),
                "official_video_frame_count": int(config.frame_count),
                "official_video_frame_size": int(config.frame_size),
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
