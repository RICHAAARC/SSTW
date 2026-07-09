"""运行 generative_video_model_probe generative video model probe readiness 框架。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.generative_video_model_probe.attack_runner import build_attack_status_records
from experiments.generative_video_model_probe.detection_runner import build_detection_records
from experiments.generative_video_model_probe.external_baseline_runner import run_external_baseline_status
from experiments.generative_video_model_probe.generation_runner import build_generation_runtime_status
from experiments.generative_video_model_probe.generalization_runner import build_generalization_status
from experiments.generative_video_model_probe.mechanism_audit import audit_mechanism
from experiments.generative_video_model_probe.table_builder import build_status_table
from main.analysis.generation_quality_audit import generation_quality_status
from main.analysis.motion_consistency_audit import motion_consistency_status
from main.analysis.semantic_consistency_audit import semantic_consistency_status
from main.generation.generation_manifest import build_generation_manifest
from main.generation.model_registry import load_generation_models, registry_snapshot
from main.generation.prompt_sampler import load_prompts
from main.generation.scheduler_adapter import load_scheduler_config
from main.generation.seed_manager import load_seeds
from main.protocol.decision import build_stage_decision
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_config() -> dict:
    return {
        "protocol": _load_json("configs/protocol/generative_video_model_probe.json"),
        "generation_models": _load_json("configs/generation/generation_models.json"),
        "prompts": _load_json("configs/generation/prompts.json"),
        "scheduler": _load_json("configs/generation/scheduler.json"),
        "seeds": _load_json("configs/generation/seeds.json"),
        "external_baselines": _load_json("configs/external_baselines/external_baselines.json"),
    }


def build_generation_records(runtime_status: dict) -> list[dict]:
    """构建生成模型状态 records, 未运行时保留 prompt、seed 和模型元数据。"""
    specs = load_generation_models("configs/generation/generation_models.json")
    prompts = load_prompts("configs/generation/prompts.json")
    seeds = load_seeds("configs/generation/seeds.json")
    scheduler = load_scheduler_config("configs/generation/scheduler.json")
    records: list[dict] = []
    for spec in specs:
        for prompt in prompts:
            for seed in seeds:
                records.append({
                    **spec.__dict__,
                    **prompt,
                    "seed_id": seed["seed_id"],
                    "scheduler_id": scheduler["scheduler_id"],
                    "trajectory_scheduler_id": scheduler["scheduler_id"],
                    "trajectory_time_grid_id": scheduler["trajectory_time_grid_id"],
                    "num_inference_steps": scheduler["num_inference_steps"],
                    "guidance_scale": scheduler["guidance_scale"],
                    "video_length_frames": 16,
                    "video_resolution": "256x256",
                    "fps": 8,
                    "heldout_prompt_status": "not_run",
                    "heldout_seed_status": "not_run",
                    "generation_model_runnable_status": runtime_status["generation_model_runnable_status"],
                    "generation_model_not_run_reason": runtime_status["generation_model_not_run_reason"],
                    "gpu_validation_status": runtime_status["gpu_validation_status"],
                    "gpu_validation_reason": runtime_status["gpu_validation_reason"],
                    "latent_capture_status": runtime_status["latent_capture_status"],
                    "latent_capture_failure_reason": runtime_status["latent_capture_failure_reason"],
                    "trajectory_capture_status": runtime_status["trajectory_capture_status"],
                    "trajectory_capture_failure_reason": runtime_status["trajectory_capture_failure_reason"],
                })
    return records


def build_quality_motion_semantic_records(generation_records: list[dict], runnable_status: str) -> list[dict]:
    """构建质量、运动和语义一致性 records。"""
    quality = generation_quality_status(runnable_status)
    motion = motion_consistency_status(runnable_status)
    semantic = semantic_consistency_status(runnable_status)
    records = []
    for item in generation_records:
        records.append({
            "generation_model_id": item["generation_model_id"],
            "prompt_id": item["prompt_id"],
            "seed_id": item["seed_id"],
            **quality,
            **motion,
            **semantic,
        })
    return records


def run(output_root: str | Path) -> dict:
    """运行 generative_video_model_probe readiness 框架并写出 governed artifacts。"""
    output_root = Path(output_root)
    config = _build_config()
    runtime_status = build_generation_runtime_status(requires_gpu=bool(config["protocol"]["requires_gpu_validation"]))
    generation_records = build_generation_records(runtime_status)
    attack_records = build_attack_status_records(runtime_status["generation_model_runnable_status"])
    event_records = build_detection_records(generation_records, attack_records)
    quality_records = build_quality_motion_semantic_records(generation_records, runtime_status["generation_model_runnable_status"])
    external_records = run_external_baseline_status("configs/external_baselines/external_baselines.json")
    generalization_status = build_generalization_status(runtime_status["generation_model_runnable_status"])
    audit = audit_mechanism(runtime_status, generalization_status)

    implementation_pass = bool(generation_records) and runtime_status["generation_model_runnable_status"] == "runnable"
    decision = build_stage_decision(implementation_pass, bool(audit["mechanism_pass"]), audit)
    decision["stage_id"] = "generative_video_model_probe"

    records_dir = output_root / "records"
    artifacts_dir = output_root / "artifacts"
    reports_dir = output_root / "reports"
    tables_dir = output_root / "tables"
    thresholds_dir = output_root / "thresholds"

    generation_path = records_dir / "generation_records.jsonl"
    event_path = records_dir / "event_scores.jsonl"
    quality_path = records_dir / "quality_motion_semantic_records.jsonl"
    external_path = records_dir / "external_baseline_records.jsonl"
    attack_path = records_dir / "attack_status_records.jsonl"
    decision_path = artifacts_dir / "generative_video_model_decision.json"
    manifest_path = artifacts_dir / "generation_manifest.json"
    registry_path = artifacts_dir / "model_registry_snapshot.json"
    status_table_path = tables_dir / "generation_model_status_table.csv"
    threshold_path = thresholds_dir / "thresholds.json"

    write_jsonl(generation_path, generation_records)
    write_jsonl(event_path, event_records)
    write_jsonl(quality_path, quality_records)
    write_jsonl(external_path, external_records)
    write_jsonl(attack_path, attack_records)
    write_json(decision_path, decision)
    write_json(registry_path, registry_snapshot(load_generation_models("configs/generation/generation_models.json")))
    write_json(threshold_path, {"threshold_status": "not_run", "threshold_not_run_reason": "generation_model_not_runnable"})
    write_csv(status_table_path, build_status_table(decision))
    manifest = build_generation_manifest(
        config_digest="local_config_snapshot",
        input_paths=["configs/protocol/generative_video_model_probe.json", "configs/generation/generation_models.json"],
        output_paths=[str(generation_path), str(event_path), str(decision_path)],
        status={"implementation_decision": decision["implementation_decision"], "mechanism_decision": decision["mechanism_decision"], **audit},
    )
    write_json(manifest_path, manifest)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "generative_video_model_report.md").write_text("# Generative Video Model Report\n\ngenerative_video_model_probe 真实生成视频模型验证未运行。原因记录在 decision 与 generation records 中。\n", encoding="utf-8")
    (reports_dir / "external_baseline_report.md").write_text("# External Baseline Report\n\n外部 baseline 已切换为显式同步机制适配器。当前只记录可运行入口状态, 尚未生成正式对比 records, 因此不用于正向 claim。\n", encoding="utf-8")
    (reports_dir / "quality_motion_semantic_report.md").write_text("# Quality Motion Semantic Report\n\n质量、运动和语义一致性指标因生成模型不可运行而标记为 not_run。\n", encoding="utf-8")
    (reports_dir / "mechanism_audit_report.md").write_text("# Mechanism Audit Report\n\n" + json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_root": str(output_root),
        "generation_record_count": len(generation_records),
        "event_record_count": len(event_records),
        "quality_record_count": len(quality_records),
        "external_baseline_record_count": len(external_records),
        "implementation_decision": decision["implementation_decision"],
        "mechanism_decision": decision["mechanism_decision"],
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 generative_video_model_probe generative video model probe readiness 框架。")
    parser.add_argument("--output-root", default="outputs/runs/generative_video_model_probe")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
