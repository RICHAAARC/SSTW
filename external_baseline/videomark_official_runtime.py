"""VideoMark 官方 PRC 生成、攻击、反演与时间匹配运行器。

该模块不复用 SSTW 的水印视频或检测分数。它以同一 prompt / seed comparison
unit 独立生成 VideoMark watermarked 和 clean-negative 视频, 施加项目共享攻击,
再调用 VideoMark 官方 PRC 检测、DDIM inversion 与 Temporal Matching Module 原语。
输出仅是 project-owned official bundle, 后续由统一 adapter 转写正式 records。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any, Iterator, Mapping

from evaluation.attacks.video_runtime_attack_protocol import apply_runtime_attack_to_frames
from external_baseline.official_eval_adapters.common import (
    REPOSITORY_GENERATED_OFFICIAL_PROVENANCE,
    build_official_reference_bundle_execution_status,
)
from external_baseline.official_runtime_progress import emit_official_reference_plan
from external_baseline.runtime_trace_io import comparable_detection_records
from external_baseline.score_semantics import official_score_formal_comparison_summary
from external_baseline.video_tensor_io import read_video_tchw_uint8, write_video_tchw
from runtime.core.digest import build_stable_digest
from runtime.core.progress import ProgressReporter, emit_progress_event


BASELINE_ID = "videomark"
DEFAULT_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"


@dataclass(frozen=True)
class VideoMarkOfficialRuntimeConfig:
    """描述一次 VideoMark official bundle 生成任务。"""

    run_root: str
    bundle_root: str
    source_dir: str
    prompt_suite_path: str
    repo_root: str = "."
    resource_root: str = ""
    model_id: str = DEFAULT_MODEL_ID
    device: str | None = None
    max_records: int | None = None
    latent_height: int = 64
    latent_width: int = 64
    num_frames: int = 16
    num_bits: int = 512
    num_inference_steps: int = 50
    num_inversion_steps: int = 50
    fps: float = 8.0
    detector_false_positive_rate: float = 0.01
    detection_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容配置。"""

        return asdict(self)


def _read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 对象, 并兼容 UTF-8 BOM。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"json_payload_must_be_object:{path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写出稳定 JSON artifact。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_token(value: Any) -> str:
    """把 comparison unit 字段转换为文件名安全 token。"""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return text.strip("_") or "unknown"


def _bundle_record_path(bundle_root: Path, record: Mapping[str, Any]) -> Path:
    """构造单条 official bundle JSON 路径。"""

    return (
        bundle_root
        / BASELINE_ID
        / "records"
        / (
            f"{_safe_token(record.get('prompt_id'))}__"
            f"{_safe_token(record.get('seed_id'))}__"
            f"{_safe_token(record.get('attack_name'))}.json"
        )
    )


def _drive_project_root_from_run_root(run_root: Path) -> Path:
    """从统一 run_root 推断项目数据根目录。"""

    parts = list(run_root.parts)
    if "runs" in parts:
        return Path(*parts[: parts.index("runs")])
    return run_root.parents[1] if len(run_root.parents) >= 2 else run_root.parent


def _default_prompt_suite_path(run_root: Path) -> Path:
    """推断 prompt / seed suite 的默认位置。"""

    return (
        _drive_project_root_from_run_root(run_root)
        / "datasets"
        / "generative_video_prompt_suite"
        / "prompt_seed_suite.json"
    )


def _prompt_seed_maps(prompt_suite_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    """读取 prompt 文本与整数 seed 映射。"""

    suite = _read_json(prompt_suite_path)
    prompts = {
        str(row["prompt_id"]): str(row["prompt_text"])
        for row in suite.get("prompts", [])
        if isinstance(row, Mapping) and row.get("prompt_id") and row.get("prompt_text")
    }
    seeds = {
        str(row["seed_id"]): int(row["seed_value"])
        for row in suite.get("seeds", [])
        if isinstance(row, Mapping)
        and row.get("seed_id")
        and row.get("seed_value") is not None
    }
    return prompts, seeds


def _stable_seed(record: Mapping[str, Any], seed_value: int, role: str) -> int:
    """为正样本和 clean negative 派生可复现实验种子。"""

    digest = build_stable_digest(
        {
            "baseline_id": BASELINE_ID,
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "seed_value": int(seed_value),
            "role": role,
        }
    )
    return int(digest[:8], 16) % 2_147_483_647


def _message_shift(record: Mapping[str, Any], sequence_length: int, frame_count: int) -> int:
    """确定当前视频使用的官方 PRC 消息窗口起点。"""

    if sequence_length <= frame_count:
        return 0
    digest = build_stable_digest(
        {
            "baseline_id": BASELINE_ID,
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "message_window": "official_prc_sequence",
        }
    )
    return int(digest[:12], 16) % (sequence_length - frame_count)


@contextmanager
def _official_source_context(source_dir: Path) -> Iterator[None]:
    """临时切换到官方源码目录, 保留其相对路径导入语义。"""

    source_text = str(source_dir.resolve())
    old_cwd = Path.cwd()
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    os.chdir(source_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted and source_text in sys.path:
            sys.path.remove(source_text)


def _load_official_backend(config: VideoMarkOfficialRuntimeConfig) -> dict[str, Any]:
    """加载 VideoMark 官方 PRC 原语与 ModelScope video pipeline。"""

    import numpy as np
    import torch
    from diffusers import TextToVideoSDPipeline
    from diffusers.schedulers import DDIMInverseScheduler
    from Levenshtein import distance

    source_dir = Path(config.source_dir).resolve()
    required = (
        source_dir / "embedding_and_extraction.py",
        source_dir / "src" / "prc.py",
        source_dir / "src" / "pseudogaussians.py",
        source_dir
        / "keys"
        / f"{config.latent_height}_{config.latent_width}_{config.num_bits}bit.pkl",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"videomark_official_source_files_missing:{missing}")
    with _official_source_context(source_dir):
        from src.prc import Decode, Detect, Encode  # type: ignore
        import src.pseudogaussians as pseudogaussians  # type: ignore
        from utils import get_video_latents, transform_video  # type: ignore

        with (source_dir / "keys" / f"{config.latent_height}_{config.latent_width}_{config.num_bits}bit.pkl").open("rb") as handle:
            encoding_key, decoding_key = pickle.load(handle)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not str(device).startswith("cuda"):
        raise RuntimeError("videomark_official_runtime_requires_cuda")
    emit_progress_event(
        "official_reference_model_load:videomark",
        f"start | model={config.model_id} device={device}",
    )
    pipeline = TextToVideoSDPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
    ).to(device)
    pipeline.safety_checker = None
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        config.model_id,
        subfolder="scheduler",
    )
    message_sequence = np.random.RandomState(11111).randint(
        0,
        2,
        size=(500, config.num_bits),
    )
    emit_progress_event("official_reference_model_load:videomark", "finish")
    return {
        "torch": torch,
        "numpy": np,
        "pipeline": pipeline,
        "generation_scheduler": pipeline.scheduler,
        "inverse_scheduler": inverse_scheduler,
        "encoding_key": encoding_key,
        "decoding_key": decoding_key,
        "Encode": Encode,
        "Detect": Detect,
        "Decode": Decode,
        "pseudogaussians": pseudogaussians,
        "get_video_latents": get_video_latents,
        "transform_video": transform_video,
        "levenshtein_distance": distance,
        "message_sequence": message_sequence,
        "device": device,
    }


def _generate_reference_pair(
    backend: Mapping[str, Any],
    config: VideoMarkOfficialRuntimeConfig,
    record: Mapping[str, Any],
    prompt_text: str,
    seed_value: int,
) -> tuple[Any, Any, int]:
    """按官方符号 PRC 构造正样本和分布匹配 clean negative。"""

    torch = backend["torch"]
    pipeline = backend["pipeline"]
    device = backend["device"]
    message_sequence = backend["message_sequence"]
    shift = _message_shift(record, len(message_sequence), config.num_frames)
    message_window = message_sequence[shift : shift + config.num_frames]
    codewords = torch.stack(
        [
            backend["Encode"](
                backend["encoding_key"],
                message=message_window[index],
            ).to(device)
            for index in range(config.num_frames)
        ]
    )
    positive_generator = torch.Generator(device=device)
    positive_generator.manual_seed(_stable_seed(record, seed_value, "watermarked"))
    positive_noise = torch.randn(
        codewords.shape,
        generator=positive_generator,
        device=device,
        dtype=codewords.dtype,
    )
    watermarked_latents = (
        codewords * positive_noise.abs()
    ).reshape(
        config.num_frames,
        1,
        4,
        config.latent_height,
        config.latent_width,
    ).to(dtype=torch.float16).permute(1, 2, 0, 3, 4)

    negative_generator = torch.Generator(device=device)
    negative_generator.manual_seed(_stable_seed(record, seed_value, "clean_negative"))
    clean_latents = torch.randn(
        (1, 4, config.num_frames, config.latent_height, config.latent_width),
        generator=negative_generator,
        device=device,
        dtype=torch.float16,
    )
    pipeline.scheduler = backend["generation_scheduler"]
    common = {
        "prompt": prompt_text,
        "num_frames": config.num_frames,
        "height": config.latent_height * 8,
        "width": config.latent_width * 8,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": 9.0,
        "output_type": "np",
    }
    with torch.no_grad():
        watermarked_frames = pipeline(
            **common,
            latents=watermarked_latents,
        ).frames[0]
        clean_frames = pipeline(
            **common,
            latents=clean_latents,
        ).frames[0]
    return watermarked_frames, clean_frames, shift


def _attack_roundtrip(
    frames: Any,
    attack_name: str,
    output_path: Path,
    fps: float,
) -> tuple[list[Any], dict[str, Any]]:
    """施加共享 runtime attack, 经 mp4 编解码后返回 detector 输入。"""

    import numpy as np
    import torch

    source_frames = [
        np.clip(np.asarray(frame) * 255.0, 0, 255).round().astype(np.uint8)
        for frame in frames
    ]
    attacked_frames, attack_metadata = apply_runtime_attack_to_frames(
        source_frames,
        attack_name,
    )
    tensor = torch.from_numpy(np.stack(attacked_frames)).permute(0, 3, 1, 2).contiguous()
    write_video_tchw(output_path, tensor, fps=fps)
    decoded, read_info = read_video_tchw_uint8(
        output_path,
        empty_error="videomark_attacked_video_empty_after_reencode",
    )
    decoded_frames = [
        frame.permute(1, 2, 0).cpu().numpy()
        for frame in decoded
    ]
    return decoded_frames, {**attack_metadata, **read_info}


def _detect_payload(
    backend: Mapping[str, Any],
    config: VideoMarkOfficialRuntimeConfig,
    frames: list[Any],
) -> dict[str, Any]:
    """执行官方 DDIM inversion、PRC Detect / Decode 和时间匹配。"""

    torch = backend["torch"]
    numpy = backend["numpy"]
    pipeline = backend["pipeline"]
    device = backend["device"]
    if not frames:
        raise RuntimeError("videomark_detection_frames_empty")
    video = backend["transform_video"](frames).to(
        pipeline.vae.dtype
    ).to(device)
    latents = backend["get_video_latents"](
        pipeline.vae,
        video,
        sample=False,
        permute=True,
    )
    pipeline.scheduler = backend["inverse_scheduler"]
    with torch.no_grad():
        reversed_latents = pipeline(
            prompt="",
            latents=latents,
            num_inference_steps=config.num_inversion_steps,
            guidance_scale=1.0,
            output_type="latent",
        ).frames
    reversed_cpu = reversed_latents.detach().cpu()
    frame_count = int(reversed_cpu.shape[2])
    indices = range(1, frame_count) if frame_count > 1 else range(frame_count)
    templates = ["".join(map(str, row.tolist())) for row in backend["message_sequence"]]
    similarities: list[float] = []
    detected_count = 0
    for frame_index in indices:
        posterior = backend["pseudogaussians"].recover_posteriors(
            reversed_cpu[:, :, frame_index].to(torch.float64).flatten(),
            variances=1.5,
        ).flatten()
        detected = bool(
            backend["Detect"](
                backend["decoding_key"],
                posterior,
                false_positive_rate=config.detector_false_positive_rate,
            )
        )
        if not detected:
            similarities.append(0.0)
            continue
        decoded = numpy.asarray(
            backend["Decode"](backend["decoding_key"], posterior)
        ).reshape(-1)
        decoded_text = "".join(map(str, decoded.astype(int).tolist()))
        minimum_distance = min(
            backend["levenshtein_distance"](decoded_text, template)
            for template in templates
        )
        similarities.append(
            max(0.0, 1.0 - float(minimum_distance) / float(config.num_bits))
        )
        detected_count += 1
    score = float(sum(similarities) / len(similarities)) if similarities else 0.0
    detector_rate = float(detected_count / len(similarities)) if similarities else 0.0
    if not math.isfinite(score):
        raise RuntimeError("videomark_non_finite_detector_score")
    return {
        "external_baseline_score": round(score, 8),
        "raw_detector_score": round(score, 8),
        "confidence": round(score, 8),
        "detected_frame_rate": round(detector_rate, 8),
        "detected": score >= config.detection_threshold,
        "threshold": config.detection_threshold,
        "score_semantics": "official_prc_detection_gated_temporal_matching_similarity",
        "score_orientation": "higher_is_more_watermarked",
        "official_inversion_frame_count": frame_count,
        "official_temporal_matching_evaluated_frame_count": len(similarities),
    }


def _payload_with_formal_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """补齐正样本与 clean negative 的正式比较摘要。"""

    merged = dict(payload)
    merged.update(official_score_formal_comparison_summary(merged))
    merged.update(official_score_formal_comparison_summary(merged, clean_negative=True))
    return merged


def build_default_videomark_official_config_from_env(
    *,
    run_root: str | Path,
    bundle_root: str | Path,
    source_dir: str | Path,
    repo_root: str | Path = ".",
    resource_root: str | Path = "",
    max_records: int | None = None,
) -> VideoMarkOfficialRuntimeConfig:
    """从服务器或 Colab 环境变量构造默认配置。"""

    root = Path(run_root)
    prompt_suite = os.environ.get("SSTW_VIDEOMARK_PROMPT_SUITE_PATH", "").strip()
    if not prompt_suite:
        prompt_suite = str(_default_prompt_suite_path(root))
    return VideoMarkOfficialRuntimeConfig(
        run_root=str(root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        prompt_suite_path=prompt_suite,
        repo_root=str(repo_root),
        resource_root=str(resource_root),
        model_id=os.environ.get("SSTW_VIDEOMARK_MODEL_ID", DEFAULT_MODEL_ID),
        device=os.environ.get("SSTW_VIDEOMARK_DEVICE", "").strip() or None,
        max_records=max_records,
        latent_height=int(os.environ.get("SSTW_VIDEOMARK_LATENT_HEIGHT", "64")),
        latent_width=int(os.environ.get("SSTW_VIDEOMARK_LATENT_WIDTH", "64")),
        num_frames=int(os.environ.get("SSTW_VIDEOMARK_NUM_FRAMES", "16")),
        num_bits=int(os.environ.get("SSTW_VIDEOMARK_NUM_BITS", "512")),
        num_inference_steps=int(os.environ.get("SSTW_VIDEOMARK_NUM_INFERENCE_STEPS", "50")),
        num_inversion_steps=int(os.environ.get("SSTW_VIDEOMARK_NUM_INVERSION_STEPS", "50")),
        fps=float(os.environ.get("SSTW_VIDEOMARK_FPS", "8")),
        detector_false_positive_rate=float(
            os.environ.get("SSTW_VIDEOMARK_INTERNAL_DETECTOR_FPR", "0.01")
        ),
        detection_threshold=float(
            os.environ.get("SSTW_VIDEOMARK_DETECTION_THRESHOLD", "0.5")
        ),
    )


def run_videomark_official_runtime(
    config: VideoMarkOfficialRuntimeConfig,
) -> dict[str, Any]:
    """生成当前 comparison units 的 VideoMark official bundle。"""

    run_root = Path(config.run_root)
    bundle_root = Path(config.bundle_root)
    records = comparable_detection_records(run_root)
    if config.max_records is not None:
        records = records[: int(config.max_records)]
    prompts, seeds = _prompt_seed_maps(Path(config.prompt_suite_path))
    emit_official_reference_plan(
        BASELINE_ID,
        runtime_detection_record_count=len(records),
        runtime_attack_count=len(
            {str(record.get("attack_name")) for record in records if record.get("attack_name")}
        ),
        extra="official_steps=prc_embed,modelscope_generate,project_attack,ddim_invert,temporal_match",
    )
    backend = _load_official_backend(config)
    cache: dict[tuple[str, str], tuple[Any, Any, int]] = {}
    generated = 0
    failures: list[dict[str, Any]] = []
    progress = ProgressReporter(
        "official_bundle_generation:videomark",
        len(records),
        "runtime_video",
    )
    video_dir = bundle_root / BASELINE_ID / "videos"
    manifest_path = bundle_root / BASELINE_ID / "official_reference_execution_manifest.json"
    for index, record in enumerate(records, start=1):
        progress.update(
            index,
            f"prompt={record.get('prompt_id')} seed={record.get('seed_id')} attack={record.get('attack_name')}",
        )
        output_json = _bundle_record_path(bundle_root, record)
        prompt_id = str(record.get("prompt_id") or "")
        seed_id = str(record.get("seed_id") or "")
        attack_name = str(record.get("attack_name") or "")
        try:
            if prompt_id not in prompts:
                raise KeyError(f"videomark_prompt_text_missing:{prompt_id}")
            if seed_id not in seeds:
                raise KeyError(f"videomark_seed_value_missing:{seed_id}")
            group_key = (prompt_id, seed_id)
            if group_key not in cache:
                cache[group_key] = _generate_reference_pair(
                    backend,
                    config,
                    record,
                    prompts[prompt_id],
                    seeds[seed_id],
                )
            watermarked_frames, clean_frames, shift = cache[group_key]
            stem = output_json.stem
            watermarked_path = video_dir / f"{stem}_watermarked.mp4"
            attacked_path = video_dir / f"{stem}_attacked.mp4"
            clean_path = video_dir / f"{stem}_clean_negative.mp4"
            watermarked_tensor = backend["torch"].from_numpy(
                backend["numpy"].stack(watermarked_frames)
            ).permute(0, 3, 1, 2)
            write_video_tchw(watermarked_path, watermarked_tensor, fps=config.fps)
            attacked_frames, attacked_metadata = _attack_roundtrip(
                watermarked_frames,
                attack_name,
                attacked_path,
                config.fps,
            )
            clean_attacked_frames, clean_metadata = _attack_roundtrip(
                clean_frames,
                attack_name,
                clean_path,
                config.fps,
            )
            positive = _detect_payload(backend, config, attacked_frames)
            negative = _detect_payload(backend, config, clean_attacked_frames)
            payload = _payload_with_formal_summary(
                {
                    **positive,
                    "external_baseline_clean_negative_score": negative["raw_detector_score"],
                    "external_baseline_clean_negative_score_semantics": negative["score_semantics"],
                    "external_baseline_clean_negative_video_path": str(clean_path),
                    "official_result_provenance": REPOSITORY_GENERATED_OFFICIAL_PROVENANCE,
                    "official_adapter_baseline_id": BASELINE_ID,
                    "official_baseline_id": BASELINE_ID,
                    "official_repository_url": "https://github.com/KYRIE-LI11/VideoMark",
                    "official_repository_commit": "9f8d78b73ab9f9f055651b1b4f37d68bdb05e7be",
                    "official_method_primitives": [
                        "prc_sign_coded_initial_noise",
                        "ddim_inverse_scheduler",
                        "prc_detect_decode",
                        "temporal_matching_minimum_edit_distance",
                    ],
                    "external_baseline_generation_model_id": config.model_id,
                    "external_baseline_source_video_path": str(watermarked_path),
                    "external_baseline_attacked_video_path": str(attacked_path),
                    "external_baseline_official_execution_mode": "videomark_official_prc_generation_inversion_temporal_matching",
                    "official_score_extraction_policy": "official_prc_detection_gated_temporal_matching_similarity",
                    "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                    "attack_protocol_status": "videomark_official_generate_then_project_runtime_attack",
                    "attack_name": attack_name,
                    "attack_metadata": attacked_metadata,
                    "clean_negative_attack_metadata": clean_metadata,
                    "prompt_id": prompt_id,
                    "seed_id": seed_id,
                    "trajectory_trace_id": record.get("trajectory_trace_id"),
                    "videomark_message_shift": shift,
                    "videomark_message_bit_count": config.num_bits,
                    "videomark_internal_detector_fpr": config.detector_false_positive_rate,
                    "official_execution_manifest_path": str(manifest_path),
                }
            )
            _write_json(output_json, payload)
            generated += 1
        except Exception as exc:  # pragma: no cover - 依赖官方 GPU 模型和 codec。
            failures.append(
                {
                    "baseline_id": BASELINE_ID,
                    "prompt_id": prompt_id,
                    "seed_id": seed_id,
                    "attack_name": attack_name,
                    "failure_reason": str(exc),
                }
            )
    progress.finish(f"generated={generated} failed={len(failures)}")
    manifest = {
        "manifest_kind": "videomark_official_reference_execution_manifest",
        "baseline_id": BASELINE_ID,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "source_dir": config.source_dir,
        "prompt_suite_path": config.prompt_suite_path,
        "official_repository_url": "https://github.com/KYRIE-LI11/VideoMark",
        "official_repository_commit": "9f8d78b73ab9f9f055651b1b4f37d68bdb05e7be",
        "runtime_config": config.to_dict(),
        "execution_status": build_official_reference_bundle_execution_status(
            generated_count=generated,
            expected_count=len(records),
            failed_count=len(failures),
        ),
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": generated,
        "failed_bundle_record_count": len(failures),
        "generated_prompt_seed_pair_count": len(cache),
        "failures": failures[:20],
        "claim_support_status": "videomark_official_bundle_generation_evidence",
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="生成 VideoMark 官方参考 bundle。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--prompt-suite-path", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--resource-root", default="")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    config = build_default_videomark_official_config_from_env(
        run_root=args.run_root,
        bundle_root=args.bundle_root,
        source_dir=args.source_dir,
        repo_root=args.repo_root,
        resource_root=args.resource_root,
        max_records=args.max_records,
    )
    if args.prompt_suite_path:
        config = VideoMarkOfficialRuntimeConfig(
            **{**config.to_dict(), "prompt_suite_path": args.prompt_suite_path}
        )
    print(
        json.dumps(
            run_videomark_official_runtime(config),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
