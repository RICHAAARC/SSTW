"""现代外部 baseline official bundle 生成与资源闭合计划。

该模块解决 Colab 冷启动中的一个实际问题: 一部分第三方 baseline 可以在当前
会话中自动下载权重并生成 official bundle, 另一部分 baseline 需要高显存、
训练得到的 extractor 或官方 maintained info。Notebook 不应让用户手动猜测
缺什么, 而应由该模块自动写出可执行计划、已生成结果和不可自动补齐的资源缺口。

重要边界: 本模块只允许调用第三方官方 API / 官方源码生成 official bundle。
它不能用 SSTW 的 `S_final`、最终判定分数、视频相似度或随机数伪造外部 baseline 分数。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from external_baseline.runtime_trace_io import comparable_detection_records
from external_baseline.score_semantics import official_score_formal_comparison_summary
from external_baseline.video_tensor_io import read_video_tchw_uint8, write_video_tchw
from external_baseline.videoseal_official_runtime import (
    ensure_videoseal_official_runtime_layout,
    videoseal_official_source_cwd,
)
from main.attacks.video_runtime_attack_protocol import apply_runtime_attack_to_video_tensor
from main.core.progress import ProgressReporter


DEFAULT_RESOURCE_REQUIREMENTS = "configs/external_baselines/official_resource_requirements.json"
MODERN_BASELINE_IDS = (
    "videoshield",
    "sigmark",
    "videomark",
    "vidsig",
    "videoseal",
)


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


def _safe_token(value: Any) -> str:
    """把 prompt、seed、attack 或 trace id 转换为 bundle 文件名 token。"""
    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _bundle_record_path(bundle_root: Path, baseline_id: str, record: Mapping[str, Any]) -> Path:
    """构造官方结果包中单条 JSON 的规范路径。"""
    prompt = _safe_token(record.get("prompt_id"))
    seed = _safe_token(record.get("seed_id"))
    attack = _safe_token(record.get("attack_name"))
    return bundle_root / baseline_id / "records" / f"{prompt}__{seed}__{attack}.json"


def _load_resource_rows(config_path: str | Path = DEFAULT_RESOURCE_REQUIREMENTS) -> dict[str, dict[str, Any]]:
    """读取现代 baseline 官方资源要求配置。"""
    config = _read_json(config_path)
    rows = config.get("resource_rows", [])
    if not isinstance(rows, list):
        raise TypeError("resource_rows_must_be_list")
    return {
        str(row.get("baseline_id")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("baseline_id")
    }


def _apply_video_tensor_attack(video: Any, attack_name: str) -> Any:
    """对官方 baseline 水印视频应用与 SSTW runtime attack 对齐的轻量文件级攻击。

    该函数只操作 official baseline 自己生成的 watermarked video, 不读取 SSTW 检测分数。
    调用方会把结果写为 mp4 并重新读取后再检测, 因此 `video_compression_runtime`
    即使在这里不裁剪帧, 也会通过 decode / re-encode 路径真实参与评分。
    """
    try:
        return apply_runtime_attack_to_video_tensor(video, attack_name)
    except ValueError as exc:
        raise ValueError(f"unsupported_videoseal_runtime_attack:{attack_name}") from exc


def _sigmoid_mean(values: Any) -> float:
    """把官方 detector 输出转换为 [0, 1] 置信度。"""
    import torch

    tensor = values.detach().float().cpu().reshape(-1)
    if tensor.numel() == 0:
        return 0.0
    if torch.all((tensor >= 0.0) & (tensor <= 1.0)):
        return float(tensor.mean().item())
    return float(torch.sigmoid(tensor).mean().item())


def _bit_accuracy(pred_bits: Any, reference_bits: Any) -> float | None:
    """计算 VideoSeal 官方 message 的 bit accuracy。"""
    if pred_bits is None or reference_bits is None:
        return None
    pred = pred_bits.detach().float().cpu().reshape(-1)
    ref = reference_bits.detach().float().cpu().reshape(-1)
    if pred.numel() == 0 or ref.numel() == 0:
        return None
    pred = (pred > 0).int()
    ref = (ref > 0).int()
    length = min(int(pred.numel()), int(ref.numel()))
    if length <= 0:
        return None
    return float((pred[:length] == ref[:length]).float().mean().item())


def _videoseal_detect_payload(
    model: Any,
    video: Any,
    *,
    reference_message: Any = None,
) -> dict[str, Any]:
    """调用 VideoSeal 官方 detector 并返回统一分数字段。

    该函数同时用于 watermarked positive 与 clean negative。这样校准分数与
    attacked positive 分数来自同一个官方 detector 输出口径。
    """

    import torch

    with torch.no_grad():
        detected_outputs = model.detect(video, is_video=True)
    preds = detected_outputs.get("preds")
    if preds is None:
        raise RuntimeError("videoseal_detect_missing_preds")
    detection_column = preds[:, 0] if preds.ndim >= 2 and preds.shape[-1] > 1 else preds
    message_logits = preds[:, 1:] if preds.ndim >= 2 and preds.shape[-1] > 1 else preds
    confidence = _sigmoid_mean(detection_column)
    bit_acc = _bit_accuracy(
        message_logits.mean(dim=0),
        reference_message[0] if reference_message is not None and reference_message.ndim >= 2 else reference_message,
    )
    return {
        "external_baseline_score": round(float(confidence), 6),
        "raw_detector_score": round(float(confidence), 6),
        "confidence": round(float(confidence), 6),
        "payload_bit_accuracy": round(float(bit_acc), 6) if bit_acc is not None else None,
        "bit_accuracy": round(float(bit_acc), 6) if bit_acc is not None else None,
        "detected": confidence >= float(os.environ.get("SSTW_VIDEOSEAL_DETECTION_THRESHOLD", "0.5")),
        "threshold": float(os.environ.get("SSTW_VIDEOSEAL_DETECTION_THRESHOLD", "0.5")),
        "score_semantics": "watermark_presence_confidence",
        "score_orientation": "higher_is_more_watermarked",
    }


def generate_videoseal_official_bundle(
    run_root: str | Path,
    bundle_root: str | Path,
    *,
    source_dir: str | Path | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """用 VideoSeal 官方 API 为当前 runtime comparison unit 生成 official bundle。

    这是当前 5 个主实验现代 baseline 中唯一可以在普通 Colab 会话中可靠自动生成的完整
    official bundle 路径。它使用 VideoSeal 官方 `videoseal.load`、`model.embed` 和
    `model.detect`, 不使用 SSTW detection score。
    """
    import torch

    source_path = Path(source_dir or os.environ.get("SSTW_VIDEOSEAL_SOURCE_DIR", "/content/SSTW/external_baseline/primary/videoseal/source"))
    source_layout_audit = ensure_videoseal_official_runtime_layout(source_path)
    sys.path.insert(0, str(source_path))
    import videoseal

    root = Path(run_root)
    bundle = Path(bundle_root)
    records = comparable_detection_records(root)
    if max_records is not None:
        records = records[:max_records]
    device = os.environ.get("SSTW_VIDEOSEAL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.environ.get("SSTW_VIDEOSEAL_MODEL_NAME", "videoseal")
    with videoseal_official_source_cwd(source_path):
        model = videoseal.load(model_name)
    model.eval()
    model.to(device)

    generated = 0
    failed: list[dict[str, Any]] = []
    progress = ProgressReporter("official_bundle_generation:videoseal", len(records), "runtime_video")
    for record in records:
        progress.update(
            generated + len(failed) + 1,
            f"prompt={record.get('prompt_id')} seed={record.get('seed_id')} attack={record.get('attack_name')}",
        )
        output_json = _bundle_record_path(bundle, "videoseal", record)
        try:
            source_video_path = Path(str(record.get("source_video_path") or ""))
            if not source_video_path.exists():
                raise FileNotFoundError(f"source_video_missing:{source_video_path}")
            video, info = read_video_tchw_uint8(source_video_path, empty_error="source_video_empty")
            if video.numel() == 0:
                raise RuntimeError("source_video_empty")
            fps = float(info.get("video_fps") or 8.0)
            video = video.float().to(device) / 255.0
            max_frames = int(os.environ.get("SSTW_VIDEOSEAL_BUNDLE_MAX_FRAMES", "0") or "0")
            if max_frames > 0:
                video = video[:max_frames]
            with torch.no_grad():
                embed_outputs = model.embed(video, is_video=True)
            watermarked = embed_outputs.get("imgs_w")
            reference_message = embed_outputs.get("msgs")
            if watermarked is None:
                raise RuntimeError("videoseal_embed_missing_imgs_w")
            attacked = _apply_video_tensor_attack(watermarked, str(record.get("attack_name") or ""))
            video_stem = output_json.stem
            baseline_video_dir = bundle / "videoseal" / "videos"
            baseline_source_video = baseline_video_dir / f"{video_stem}_watermarked.mp4"
            baseline_attacked_video = baseline_video_dir / f"{video_stem}_attacked.mp4"
            baseline_video_dir.mkdir(parents=True, exist_ok=True)
            write_video_tchw(baseline_source_video, watermarked, fps=fps)
            write_video_tchw(baseline_attacked_video, attacked, fps=fps)
            attacked_uint8, attacked_read_info = read_video_tchw_uint8(
                baseline_attacked_video,
                empty_error="videoseal_attacked_video_empty_after_reencode",
            )
            attacked_for_detection = attacked_uint8.float().to(device) / 255.0
            clean_negative = _apply_video_tensor_attack(video, str(record.get("attack_name") or ""))
            clean_negative_video = baseline_video_dir / f"{video_stem}_clean_negative.mp4"
            write_video_tchw(clean_negative_video, clean_negative, fps=fps)
            clean_negative_uint8, clean_negative_read_info = read_video_tchw_uint8(
                clean_negative_video,
                empty_error="videoseal_clean_negative_video_empty_after_reencode",
            )
            clean_negative_for_detection = clean_negative_uint8.float().to(device) / 255.0
            score_payload = _videoseal_detect_payload(model, attacked_for_detection, reference_message=reference_message)
            clean_negative_payload = _videoseal_detect_payload(model, clean_negative_for_detection)
            payload = {
                **score_payload,
                "external_baseline_clean_negative_score": clean_negative_payload["raw_detector_score"],
                "external_baseline_clean_negative_score_semantics": clean_negative_payload["score_semantics"],
                "external_baseline_clean_negative_video_path": str(clean_negative_video),
                "official_result_provenance": "repository_generated_from_third_party_official_code",
                "official_adapter_baseline_id": "videoseal",
                "official_baseline_id": "videoseal",
                "official_source_layout_status": source_layout_audit["layout_status"],
                "official_video_io_backend": info.get("video_io_backend"),
                "official_attacked_video_io_backend": attacked_read_info.get("video_io_backend"),
                "official_clean_negative_video_io_backend": clean_negative_read_info.get("video_io_backend"),
                "external_baseline_generation_model_id": "videoseal_official_api",
                "external_baseline_source_video_path": str(baseline_source_video),
                "external_baseline_attacked_video_path": str(baseline_attacked_video),
                "external_baseline_official_execution_mode": "videoseal_official_embed_detect",
                "official_score_extraction_policy": "videoseal_official_detect_presence_confidence",
                "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                "attack_protocol_status": "videoseal_official_embed_then_project_runtime_attack",
                "source_sstw_video_path": str(source_video_path),
                "sstw_attacked_video_path": str(record.get("attacked_video_path") or ""),
                "attack_name": record.get("attack_name"),
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "trajectory_trace_id": record.get("trajectory_trace_id"),
                "official_execution_manifest_path": str(bundle / "videoseal" / "official_bundle_generation_manifest.json"),
            }
            payload = {
                **payload,
                **official_score_formal_comparison_summary(payload),
                **official_score_formal_comparison_summary(payload, clean_negative=True),
            }
            _write_json(output_json, payload)
            generated += 1
        except Exception as exc:  # pragma: no cover - 依赖 Colab GPU、视频编码和官方 checkpoint
            failed.append({
                "baseline_id": "videoseal",
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "failure_reason": str(exc),
            })
    progress.finish(f"generated={generated} failed={len(failed)}")
    manifest = {
        "manifest_kind": "videoseal_official_bundle_generation_manifest",
        "baseline_id": "videoseal",
        "run_root": str(root),
        "bundle_root": str(bundle),
        "official_repository_url": "https://github.com/facebookresearch/videoseal",
        "official_api": "videoseal.load/embed/detect",
        "official_source_dir": str(source_path),
        "official_source_layout_audit": source_layout_audit,
        "official_video_io_backend": "imageio_v3",
        "official_model_name": model_name,
        "device": device,
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": generated,
        "failed_bundle_record_count": len(failed),
        "failures": failed[:20],
        "claim_support_status": "videoseal_official_bundle_generation_evidence",
    }
    _write_json(bundle / "videoseal" / "official_bundle_generation_manifest.json", manifest)
    return manifest


def build_official_bundle_generation_plan(
    run_root: str | Path,
    bundle_root: str | Path,
    *,
    resource_config_path: str | Path = DEFAULT_RESOURCE_REQUIREMENTS,
) -> dict[str, Any]:
    """构建 5 个 modern baseline 的 official bundle 自动生成计划。"""
    records = comparable_detection_records(run_root)
    rows = _load_resource_rows(resource_config_path)
    plan_rows: list[dict[str, Any]] = []
    for baseline_id in MODERN_BASELINE_IDS:
        row = rows.get(baseline_id, {})
        plan_rows.append({
            "baseline_id": baseline_id,
            "official_repository_url": row.get("official_repository_url"),
            "colab_l4_auto_bundle_status": row.get("colab_l4_auto_bundle_status"),
            "automatic_bundle_generation_supported_by_sstw": bool(row.get("automatic_bundle_generation_supported_by_sstw")),
            "strict_gate_resolution": row.get("strict_gate_resolution"),
            "required_public_or_user_resources": row.get("required_public_or_user_resources", []),
            "expected_bundle_record_count": len(records),
            "bundle_root": str(Path(bundle_root) / baseline_id),
            "resource_blocker": row.get("reason_automatic_bundle_not_supported"),
        })
    auto_supported = [row["baseline_id"] for row in plan_rows if row["automatic_bundle_generation_supported_by_sstw"]]
    auto_blocked = [row["baseline_id"] for row in plan_rows if not row["automatic_bundle_generation_supported_by_sstw"]]
    return {
        "manifest_kind": "external_baseline_official_bundle_generation_plan",
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "resource_config_path": str(resource_config_path),
        "runtime_comparison_unit_count": len(records),
        "baseline_count": len(plan_rows),
        "auto_supported_baselines": auto_supported,
        "auto_blocked_baselines": auto_blocked,
        "auto_blocked_baseline_count": len(auto_blocked),
        "plan_rows": plan_rows,
        "claim_support_status": "official_bundle_generation_plan_not_claim_evidence",
    }


def run_official_bundle_generation(
    run_root: str | Path,
    bundle_root: str | Path,
    *,
    generate_auto_supported: bool,
    resource_config_path: str | Path = DEFAULT_RESOURCE_REQUIREMENTS,
) -> dict[str, Any]:
    """执行可自动生成的 official bundle, 并写出计划和决策。"""
    root = Path(run_root)
    bundle = Path(bundle_root)
    plan = build_official_bundle_generation_plan(root, bundle, resource_config_path=resource_config_path)
    generation_manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if generate_auto_supported and "videoseal" in plan["auto_supported_baselines"]:
        try:
            generation_manifests.append(generate_videoseal_official_bundle(root, bundle))
        except Exception as exc:  # pragma: no cover - 依赖 Colab GPU 和第三方模型
            failures.append({"baseline_id": "videoseal", "failure_reason": str(exc)})
    if generate_auto_supported and "videomark" in plan["auto_supported_baselines"]:
        try:
            from external_baseline.videomark_official_runtime import (
                build_default_videomark_official_config_from_env,
                run_videomark_official_runtime,
            )

            source_dir = Path("external_baseline/primary/videomark/source")
            config = build_default_videomark_official_config_from_env(
                run_root=root,
                bundle_root=bundle,
                source_dir=source_dir,
                repo_root=".",
                max_records=None,
            )
            generation_manifests.append(run_videomark_official_runtime(config))
        except Exception as exc:  # pragma: no cover - 依赖 Colab GPU 和第三方模型
            failures.append({"baseline_id": "videomark", "failure_reason": str(exc)})
    if generate_auto_supported and "vidsig" in plan["auto_supported_baselines"]:
        try:
            from external_baseline.vidsig_official_runtime import (
                build_default_vidsig_official_config_from_env,
                run_vidsig_official_runtime,
            )

            source_dir = Path("external_baseline/primary/vidsig/source")
            config = build_default_vidsig_official_config_from_env(
                run_root=root,
                bundle_root=bundle,
                source_dir=source_dir,
                repo_root=".",
                max_records=None,
            )
            generation_manifests.append(run_vidsig_official_runtime(config))
        except Exception as exc:  # pragma: no cover - 依赖 Colab GPU 和第三方模型
            failures.append({"baseline_id": "vidsig", "failure_reason": str(exc)})
    if generate_auto_supported and "videoshield" in plan["auto_supported_baselines"]:
        try:
            from external_baseline.videoshield_official_runtime import (
                build_default_videoshield_official_config_from_env,
                run_videoshield_official_runtime,
            )

            source_dir = Path("external_baseline/primary/videoshield/source")
            config = build_default_videoshield_official_config_from_env(
                run_root=root,
                bundle_root=bundle,
                source_dir=source_dir,
                repo_root=".",
                max_records=None,
            )
            generation_manifests.append(run_videoshield_official_runtime(config))
        except Exception as exc:  # pragma: no cover - 依赖 Colab GPU 和第三方模型
            failures.append({"baseline_id": "videoshield", "failure_reason": str(exc)})
    decision = {
        "artifact_name": "external_baseline_official_bundle_generation_decision.json",
        "manifest_kind": "external_baseline_official_bundle_generation_decision",
        "run_root": str(root),
        "bundle_root": str(bundle),
        "official_bundle_generation_decision": "PASS" if not failures else "FAIL",
        "strict_gate_auto_bundle_closure": plan["auto_blocked_baseline_count"] == 0 and not failures,
        "strict_gate_auto_bundle_status": (
            "all_modern_baselines_auto_generated"
            if plan["auto_blocked_baseline_count"] == 0 and not failures
            else "manual_official_bundles_still_required_for_auto_blocked_baselines"
        ),
        "generate_auto_supported": bool(generate_auto_supported),
        "generated_baseline_count": len(generation_manifests),
        "generation_failure_count": len(failures),
        "generation_failures": failures,
        "plan_path": str(root / "artifacts" / "external_baseline_official_bundle_generation_plan.json"),
        "auto_blocked_baselines": plan["auto_blocked_baselines"],
        "auto_blocked_baseline_count": plan["auto_blocked_baseline_count"],
        "claim_support_status": "official_bundle_generation_evidence_only",
    }
    _write_json(root / "artifacts" / "external_baseline_official_bundle_generation_plan.json", plan)
    _write_json(root / "artifacts" / "external_baseline_official_bundle_generation_decision.json", decision)
    return {"plan": plan, "decision": decision, "generation_manifests": generation_manifests}


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="生成或规划现代 external baseline official bundle。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--resource-config-path", default=DEFAULT_RESOURCE_REQUIREMENTS)
    parser.add_argument("--generate-auto-supported", action="store_true")
    args = parser.parse_args()
    payload = run_official_bundle_generation(
        args.run_root,
        args.bundle_root,
        generate_auto_supported=args.generate_auto_supported,
        resource_config_path=args.resource_config_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
