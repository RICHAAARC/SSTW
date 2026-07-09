"""B5 生成视频 runtime attack runner。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from experiments.generative_video_model_probe.formal_motion_claim_filter import select_motion_claim_generation_records
from main.attacks.video_runtime_attack_protocol import (
    PAPER_PROFILE_RUNTIME_ATTACKS,
    RUNTIME_ATTACK_SPECS,
    apply_runtime_attack_to_frames,
    required_runtime_attack_names_from_config,
)
from main.core.progress import ProgressReporter
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


ATTACK_NAMES = tuple(RUNTIME_ATTACK_SPECS)
RUNTIME_PILOT_ATTACKS = PAPER_PROFILE_RUNTIME_ATTACKS
DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"


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


def _write_video_frames(video_path: Path, frames: list[Any], fps: int = 8, attack_metadata: dict | None = None) -> None:
    """把帧序列写回 mp4 文件。

    若 attack metadata 声明了 codec 或 CRF 参数, 这里会交给 imageio/ffmpeg。
    如果当前环境缺少对应编码器, 上层会把该 attack 记录为 failed, 避免将未执行
    的 H.264 / H.265 强度攻击误写成论文级证据。
    """
    import imageio.v3 as iio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"fps": fps}
    attack_metadata = attack_metadata or {}
    if attack_metadata.get("video_writer_codec"):
        kwargs["codec"] = attack_metadata["video_writer_codec"]
    if attack_metadata.get("video_writer_output_params"):
        kwargs["output_params"] = list(attack_metadata["video_writer_output_params"])
    iio.imwrite(video_path, frames, **kwargs)


def _apply_runtime_attack(frames: list[Any], attack_name: str) -> tuple[list[Any], dict]:
    """对视频帧执行 runtime attack。

    具体 attack 协议和帧级实现由 `main.attacks.video_runtime_attack_protocol`
    统一管理, 本函数保留为 runner 内部兼容入口。
    """
    return apply_runtime_attack_to_frames(frames, attack_name)


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


def _load_required_attack_names(config_path: str | Path | None, attack_names: tuple[str, ...] | None) -> tuple[str, ...]:
    """从命令行参数或 protocol config 解析本次必须执行的 attack 名称。"""

    if attack_names:
        return tuple(str(item) for item in attack_names if str(item))
    if config_path is None:
        return RUNTIME_PILOT_ATTACKS
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise TypeError(f"protocol config 顶层必须是对象: {config_path}")
    return required_runtime_attack_names_from_config(config)


def build_runtime_attack_records(
    run_root: str | Path,
    attack_names: tuple[str, ...] | None = None,
    config_path: str | Path | None = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict]:
    """对 generation records 中的真实 mp4 执行 runtime attacks 并返回 records。"""
    run_root = Path(run_root)
    selected_attack_names = _load_required_attack_names(config_path, attack_names)
    generation_records = [
        record for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success"
    ]
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    selection = select_motion_claim_generation_records(generation_records, formal_metric_records)
    records: list[dict] = []
    total_attack_jobs = len(selection.eligible_generation_records) * len(selected_attack_names)
    progress = ProgressReporter("runtime_attack_video_transform", total_attack_jobs, "attack_video")
    progress_index = 0
    for generation_record in selection.eligible_generation_records:
        source_video_path = _resolve_video_path(run_root, generation_record)
        for attack_name in selected_attack_names:
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
                _write_video_frames(attacked_video_path, attacked_frames, attack_metadata=attack_metadata)
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


def audit_runtime_attack_records(
    records: list[dict],
    run_root: str | Path | None = None,
    required_attack_names: tuple[str, ...] | None = None,
) -> dict:
    """审计 runtime attack records 的覆盖情况。"""
    ready_records = [record for record in records if record.get("attack_runtime_status") == "ready"]
    attack_names = {str(record.get("attack_name")) for record in ready_records if record.get("attack_name")}
    required_attack_set = {str(name) for name in (required_attack_names or ()) if str(name)}
    missing_required_attack_names = sorted(required_attack_set - attack_names)
    selection_fields: dict = {}
    if run_root is not None:
        root = Path(run_root)
        generation_records = _read_jsonl(root / "records" / "generation_records.jsonl")
        formal_metric_records = _read_jsonl(root / "records" / "formal_quality_motion_semantic_records.jsonl")
        selection_fields = select_motion_claim_generation_records(generation_records, formal_metric_records).audit_fields()
    return {
        "stage_id": "generative_video_runtime_attack_runner",
        "runtime_attack_decision": "PASS"
        if ready_records and len(ready_records) == len(records) and not missing_required_attack_names
        else "FAIL",
        "runtime_attack_record_count": len(records),
        "runtime_attack_ready_count": len(ready_records),
        "runtime_attack_count": len(attack_names),
        "runtime_attack_names": sorted(attack_names),
        "required_runtime_attack_names": sorted(required_attack_set),
        "missing_required_runtime_attack_names": missing_required_attack_names,
        "missing_required_runtime_attack_count": len(missing_required_attack_names),
        "attack_matrix_evidence_level": "runtime_video_file",
        "claim_support_status": "runtime_attack_evidence_only",
        **selection_fields,
    }


def run_runtime_attacks(
    run_root: str | Path,
    attack_names: tuple[str, ...] | None = None,
    config_path: str | Path | None = DEFAULT_PROTOCOL_CONFIG,
) -> dict:
    """执行 runtime attacks 并写出 governed records、table、decision 和 report。"""
    run_root = Path(run_root)
    selected_attack_names = _load_required_attack_names(config_path, attack_names)
    records = build_runtime_attack_records(run_root, attack_names=selected_attack_names, config_path=None)
    audit = audit_runtime_attack_records(records, run_root, required_attack_names=selected_attack_names)
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
        f"- required_runtime_attack_names: {', '.join(audit['required_runtime_attack_names'])}\n"
        f"- missing_required_runtime_attack_names: {', '.join(audit['missing_required_runtime_attack_names']) if audit['missing_required_runtime_attack_names'] else 'none'}\n"
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
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    parser.add_argument("--attacks", nargs="*", default=None)
    args = parser.parse_args()
    payload = run_runtime_attacks(
        args.run_root,
        attack_names=tuple(args.attacks) if args.attacks else None,
        config_path=args.config_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
