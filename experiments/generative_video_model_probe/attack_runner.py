"""B5 生成视频 runtime attack runner。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from experiments.generative_video_model_probe.formal_motion_claim_filter import select_motion_claim_generation_records
from main.core.progress import ProgressReporter
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


ATTACK_NAMES = (
    "no_attack", "h264_compression", "h265_compression", "spatial_resize", "crop_resize",
    "temporal_crop", "local_clip", "regular_frame_dropping", "irregular_frame_dropping",
    "frame_duplication", "speed_change", "frame_rate_resampling", "gaussian_noise", "blur",
)

RUNTIME_PILOT_ATTACKS = (
    "video_compression_runtime",
    "temporal_crop_runtime",
    "frame_rate_resampling_runtime",
)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    """计算文件 sha256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_token(value: Any) -> str:
    """把模型、prompt、seed 或 attack 名称转换为安全文件名片段。"""
    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _resolve_video_path(run_root: Path, generation_record: dict) -> Path:
    """解析 generation record 中登记的视频路径。"""
    raw_path = Path(str(generation_record.get("video_path") or ""))
    if raw_path.exists():
        return raw_path
    return run_root / "videos" / raw_path.name


def _load_video_frames(video_path: Path, max_frames: int | None = None) -> list[Any]:
    """从 mp4 文件读取帧。

    该函数属于通用工程写法, 用于将运行时攻击限制在文件级视频变换上。它不会读取或修改 latent records。
    """
    import imageio.v3 as iio

    frames: list[Any] = []
    for frame_index, frame in enumerate(iio.imiter(video_path)):
        if max_frames is not None and frame_index >= max_frames:
            break
        frames.append(frame)
    return frames


def _write_video_frames(video_path: Path, frames: list[Any], fps: int = 8) -> None:
    """把帧序列写回 mp4 文件。"""
    import imageio.v3 as iio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(video_path, frames, fps=fps)


def _apply_runtime_attack(frames: list[Any], attack_name: str) -> tuple[list[Any], dict]:
    """对视频帧执行 runtime attack。

    当前实现用于 small-scale pilot 工程验证, 攻击实际作用在 mp4 解码帧上。它不是最终论文级攻击强度定义,
    但比 proxy_postprocess 更接近真实 runtime 路径。
    """
    if not frames:
        raise ValueError("no_decodable_frames")
    if attack_name == "video_compression_runtime":
        return list(frames), {
            "attack_transform": "decode_reencode",
            "attack_strength": "runtime_reencode_default_quality",
            "runtime_attack_expected_effect": "codec_quantization_or_container_rewrite",
        }
    if attack_name == "temporal_crop_runtime":
        if len(frames) >= 4:
            attacked = frames[1:-1]
        else:
            attacked = list(frames)
        return attacked, {
            "attack_transform": "drop_first_and_last_frame_when_possible",
            "attack_strength": "crop_boundary_frames",
            "runtime_attack_expected_effect": "temporal_boundary_shift",
        }
    if attack_name == "frame_rate_resampling_runtime":
        if len(frames) >= 3:
            attacked = frames[::2]
        else:
            attacked = list(frames)
        return attacked, {
            "attack_transform": "keep_every_second_frame_when_possible",
            "attack_strength": "fps_downsample_by_2_proxy",
            "runtime_attack_expected_effect": "time_grid_resampling",
        }
    raise ValueError(f"unsupported_runtime_attack:{attack_name}")


def _build_attacked_video_path(run_root: Path, generation_record: dict, attack_name: str) -> Path:
    """构造 attacked video 的落盘路径。"""
    stem = "_".join([
        _safe_token(generation_record.get("generation_model_id")),
        _safe_token(generation_record.get("prompt_id")),
        _safe_token(generation_record.get("seed_id")),
        _safe_token(attack_name),
    ])
    return run_root / "attacked_videos" / f"{stem}.mp4"


def build_attack_status_records(runnable_status: str) -> list[dict]:
    """生成攻击矩阵状态记录, 未生成视频时只记录 not_run。"""
    records = []
    for attack_name in ATTACK_NAMES:
        records.append({
            "attack_name": attack_name,
            "attack_failure_status": "not_run" if runnable_status != "runnable" else "pending_runtime",
            "attack_failure_reason": "generation_model_not_runnable" if runnable_status != "runnable" else "none",
        })
    return records


def build_runtime_attack_records(run_root: str | Path, attack_names: tuple[str, ...] = RUNTIME_PILOT_ATTACKS) -> list[dict]:
    """对 generation records 中的真实 mp4 执行 runtime attacks 并返回 records。"""
    run_root = Path(run_root)
    generation_records = [
        record for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success"
    ]
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    selection = select_motion_claim_generation_records(generation_records, formal_metric_records)
    records: list[dict] = []
    total_attack_jobs = len(selection.eligible_generation_records) * len(attack_names)
    progress = ProgressReporter("runtime_attack_video_transform", total_attack_jobs, "attack_video")
    progress_index = 0
    for generation_record in selection.eligible_generation_records:
        source_video_path = _resolve_video_path(run_root, generation_record)
        for attack_name in attack_names:
            progress_index += 1
            progress.update(
                progress_index,
                f"prompt={generation_record.get('prompt_id')} seed={generation_record.get('seed_id')} attack={attack_name}",
            )
            record = with_flow_evidence_protocol_defaults({
                "record_version": "generative_video_runtime_attack_v1",
                "generation_model_id": generation_record.get("generation_model_id"),
                "prompt_id": generation_record.get("prompt_id"),
                "seed_id": generation_record.get("seed_id"),
                "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
                "attack_name": attack_name,
                "attack_matrix_evidence_level": "runtime_video_file",
                "attack_runtime_status": "failed",
                "attack_runtime_failure_reason": "not_run",
                "source_video_path": str(source_video_path),
                "source_video_sha256": None,
                "attacked_video_path": None,
                "attacked_video_sha256": None,
                "source_frame_count": 0,
                "attacked_frame_count": 0,
                "claim_support_status": "runtime_attack_evidence_only",
            },
                negative_family=None,
                trajectory_source_level="runtime_video_file_attack",
                flow_state_admissibility_status="not_evaluated",
                claim_support_status="runtime_attack_evidence_only",
            )
            try:
                if not source_video_path.exists():
                    raise FileNotFoundError("source_video_not_found")
                frames = _load_video_frames(source_video_path)
                attacked_frames, attack_metadata = _apply_runtime_attack(frames, attack_name)
                attacked_video_path = _build_attacked_video_path(run_root, generation_record, attack_name)
                _write_video_frames(attacked_video_path, attacked_frames)
                record.update({
                    "attack_runtime_status": "ready",
                    "attack_runtime_failure_reason": "none",
                    "source_video_sha256": _sha256_file(source_video_path),
                    "attacked_video_path": str(attacked_video_path),
                    "attacked_video_sha256": _sha256_file(attacked_video_path),
                    "source_frame_count": len(frames),
                    "attacked_frame_count": len(attacked_frames),
                    **attack_metadata,
                })
            except Exception as exc:  # pragma: no cover - 依赖具体视频解码和编码后端
                record.update({
                    "attack_runtime_status": "failed",
                    "attack_runtime_failure_reason": str(exc),
                })
            records.append(record)
    ready_count = sum(1 for record in records if record.get("attack_runtime_status") == "ready")
    progress.finish(f"ready={ready_count} failed={len(records) - ready_count}")
    return records


def audit_runtime_attack_records(records: list[dict], run_root: str | Path | None = None) -> dict:
    """审计 runtime attack records 的覆盖情况。"""
    ready_records = [record for record in records if record.get("attack_runtime_status") == "ready"]
    attack_names = {str(record.get("attack_name")) for record in ready_records if record.get("attack_name")}
    selection_fields: dict = {}
    if run_root is not None:
        root = Path(run_root)
        generation_records = _read_jsonl(root / "records" / "generation_records.jsonl")
        formal_metric_records = _read_jsonl(root / "records" / "formal_quality_motion_semantic_records.jsonl")
        selection_fields = select_motion_claim_generation_records(generation_records, formal_metric_records).audit_fields()
    return {
        "stage_id": "generative_video_runtime_attack_runner",
        "runtime_attack_decision": "PASS" if ready_records and len(ready_records) == len(records) else "FAIL",
        "runtime_attack_record_count": len(records),
        "runtime_attack_ready_count": len(ready_records),
        "runtime_attack_count": len(attack_names),
        "attack_matrix_evidence_level": "runtime_video_file",
        "claim_support_status": "runtime_attack_evidence_only",
        **selection_fields,
    }


def run_runtime_attacks(run_root: str | Path, attack_names: tuple[str, ...] = RUNTIME_PILOT_ATTACKS) -> dict:
    """执行 runtime attacks 并写出 governed records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_runtime_attack_records(run_root, attack_names=attack_names)
    audit = audit_runtime_attack_records(records, run_root)
    write_jsonl(run_root / "records" / "runtime_attack_records.jsonl", records)
    write_csv(run_root / "tables" / "runtime_attack_table.csv", records)
    write_json(run_root / "artifacts" / "runtime_attack_decision.json", audit)
    report = (
        "# Runtime Attack Runner Report\n\n"
        "该报告记录对真实 mp4 文件执行的 runtime attack。当前结果用于工程验证 attack runner 与 attacked video 落盘链路, "
        "不能单独支撑最终论文 claim。\n\n"
        f"- runtime_attack_decision: {audit['runtime_attack_decision']}\n"
        f"- runtime_attack_record_count: {audit['runtime_attack_record_count']}\n"
        f"- runtime_attack_ready_count: {audit['runtime_attack_ready_count']}\n"
        f"- runtime_attack_count: {audit['runtime_attack_count']}\n"
        f"- motion_claim_eligible_generation_count: {audit.get('motion_claim_eligible_generation_count')}\n"
        f"- motion_claim_excluded_generation_count: {audit.get('motion_claim_excluded_generation_count')}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "runtime_attack_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="对生成视频执行 runtime 文件级攻击。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--attacks", nargs="*", default=list(RUNTIME_PILOT_ATTACKS))
    args = parser.parse_args()
    payload = run_runtime_attacks(args.run_root, attack_names=tuple(args.attacks))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
