"""验证 SSTW 与正式 baseline 使用同口径配对视频质量。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.generative_video_model_probe.paper_result_artifact_builders as builders
from evaluation.protocol.record_writer import read_jsonl, write_jsonl


BASELINES = ("videoshield", "vidsig", "videoseal", "videomark", "wam_frame")
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"


def _context() -> dict:
    """构造与三个正式 profile 公共质量契约一致的轻量上下文。"""

    return {
        "paper_result_level": "probe_paper",
        "target_fpr": 0.1,
        "required_modern_external_baseline_adapter_names": list(BASELINES),
        "require_baseline_matched_video_quality_metrics": True,
        "video_quality_comparison_protocol": (
            "same_model_prompt_seed_clean_reference_to_method_own_watermarked_source_"
            "paired_psnr_ssim_temporal_delta"
        ),
    }


def _write_quality_sources(run_root: Path, *, missing_baseline_source: str | None = None) -> None:
    """写入1个 SSTW 配对记录、5个 baseline source 锚点与公平鲁棒性记录。"""

    reference = run_root / "videos" / "clean.mp4"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"clean")
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", [{
        "method_variant": "sstw_full_method",
        "sample_role": "attacked_positive_source",
        "paired_video_quality_required": True,
        "paired_video_quality_status": "ready",
        "paired_watermark_psnr": 45.0,
        "paired_watermark_ssim": 0.995,
        "paired_temporal_delta_error": 0.002,
        "formal_metric_result_used_for_claim": True,
    }])
    external_rows: list[dict] = []
    for index, baseline_id in enumerate(BASELINES):
        source = run_root / "baseline_sources" / f"{baseline_id}.mp4"
        if baseline_id != missing_baseline_source:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"watermarked")
        for attack_name in ("video_compression_runtime", "temporal_crop_runtime"):
            external_rows.append({
                "external_baseline_name": baseline_id,
                "metric_status": "measured_formal",
                "external_baseline_result_used_for_claim": True,
                "generation_model_id": "model",
                "prompt_id": "prompt",
                "seed_id": "seed",
                "attack_name": attack_name,
                "source_video_path": str(reference),
                "baseline_clean_reference_video_path": str(reference),
                "baseline_clean_reference_status": (
                    "matched_same_model_prompt_seed_clean_reference"
                ),
                "baseline_input_source_policy": (
                    "baseline_embeds_own_watermark_into_clean_reference"
                ),
                "external_baseline_source_video_path": str(source),
            })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", external_rows)
    fair_rows = []
    for index, method_id in enumerate((SSTW_METHOD_ID, *BASELINES)):
        fair_rows.append({
            "method_id": method_id,
            "method_role": "proposed_method" if method_id == SSTW_METHOD_ID else "modern_external_baseline",
            "target_fpr": 0.1,
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "tpr_at_target_fpr": 0.9 - index * 0.05,
            "tpr_ci_lower": 0.8 - index * 0.05,
            "tpr_ci_upper": 0.95 - index * 0.05,
        })
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", fair_rows)


@pytest.mark.quick
def test_quality_builder_uses_sstw_pairs_and_each_baseline_own_watermarked_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5个 baseline 必须各自用匹配 clean reference 计算 PSNR、SSIM 与时间差分。"""

    run_root = tmp_path / "run"
    _write_quality_sources(run_root)
    calls: list[tuple[Path, Path]] = []

    def fake_paired_metrics(reference: Path, candidate: Path) -> dict:
        calls.append((Path(reference), Path(candidate)))
        baseline_index = BASELINES.index(Path(candidate).stem)
        return {
            "paired_video_quality_status": "ready",
            "paired_watermark_psnr": 40.0 - baseline_index,
            "paired_watermark_ssim": 0.98 - baseline_index * 0.01,
            "paired_temporal_delta_error": 0.01 + baseline_index * 0.001,
        }

    monkeypatch.setattr(builders, "compute_paired_video_quality_metrics", fake_paired_metrics)
    records = builders.build_video_quality_metric_records(run_root, _context())
    audit = builders.audit_video_quality_metric_records(records, _context())

    assert audit["video_quality_metric_decision"] == "PASS"
    assert audit["sstw_paired_video_quality_ready"] is True
    assert audit["baseline_matched_video_quality_ready"] is True
    assert {record["method_id"] for record in records} == {SSTW_METHOD_ID, *BASELINES}
    assert len(calls) == len(BASELINES)
    assert all(reference.name == "clean.mp4" for reference, _ in calls)
    baseline_records = {
        record["method_id"]: record
        for record in records
        if record["method_role"] == "modern_external_baseline"
    }
    assert baseline_records["videoshield"]["paired_quality_unit_count"] == 1
    assert baseline_records["videoshield"]["quality_metric_source_record_count"] == 2
    assert baseline_records["videoshield"]["mean_paired_watermark_psnr"] == 40.0
    assert baseline_records["videoshield"]["mean_paired_watermark_ssim"] == 0.98
    assert baseline_records["videoshield"]["mean_paired_temporal_delta_error"] == 0.01
    sstw = next(record for record in records if record["method_id"] == SSTW_METHOD_ID)
    assert sstw["mean_paired_watermark_psnr"] == 45.0
    assert sstw["mean_paired_watermark_ssim"] == 0.995
    assert sstw["mean_paired_temporal_delta_error"] == 0.002


@pytest.mark.quick
def test_quality_builder_fails_closed_when_any_required_baseline_video_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一 baseline 缺少自身 watermarked source 时不得生成 ready 质量结论。"""

    run_root = tmp_path / "run"
    _write_quality_sources(run_root, missing_baseline_source="videoseal")
    monkeypatch.setattr(builders, "compute_paired_video_quality_metrics", lambda *_: {
        "paired_video_quality_status": "ready",
        "paired_watermark_psnr": 40.0,
        "paired_watermark_ssim": 0.98,
        "paired_temporal_delta_error": 0.01,
    })

    records = builders.build_video_quality_metric_records(run_root, _context())
    audit = builders.audit_video_quality_metric_records(records, _context())
    videoseal = next(record for record in records if record["method_id"] == "videoseal")

    assert videoseal["video_quality_metric_status"] == "blocked"
    assert "baseline_watermarked_source_video_missing" in videoseal["quality_metric_failure_reasons"]
    assert audit["video_quality_metric_decision"] == "FAIL"
    assert audit["baseline_matched_video_quality_ready"] is False
    assert audit["baseline_matched_video_quality_missing_method_ids"] == ["videoseal"]


@pytest.mark.quick
def test_quality_robustness_figure_encodes_real_paired_psnr_and_ssim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """质量—鲁棒性图必须使用配对 PSNR/SSIM 与固定 FPR TPR, 禁止使用 ready count。"""

    run_root = tmp_path / "run"
    _write_quality_sources(run_root)
    monkeypatch.setattr(builders, "compute_paired_video_quality_metrics", lambda *_: {
        "paired_video_quality_status": "ready",
        "paired_watermark_psnr": 40.0,
        "paired_watermark_ssim": 0.98,
        "paired_temporal_delta_error": 0.01,
    })

    builders.run_paper_result_artifact_builders(
        run_root,
        "configs/protocol/probe_paper_generative_probe.json",
    )
    figure = json.loads(
        (run_root / "figures" / "video_quality_robustness_tradeoff_figure.json").read_text(
            encoding="utf-8"
        )
    )
    records = read_jsonl(run_root / "records" / "video_quality_metric_records.jsonl")

    assert figure["encoding"] == {
        "x": "mean_paired_watermark_psnr",
        "y": "robustness_tpr_at_target_fpr",
        "color": "method_id",
    }
    assert figure["alternate_encodings"][0]["x"] == "mean_paired_watermark_ssim"
    assert len(figure["figure_rows"]) == 6
    assert all("formal_metric_ready_count" not in row for row in figure["figure_rows"])
    assert all(row["mean_paired_watermark_psnr"] is not None for row in figure["figure_rows"])
    assert all(row["mean_paired_watermark_ssim"] is not None for row in figure["figure_rows"])
    assert all(record["video_quality_metric_status"] == "ready" for record in records)

@pytest.mark.quick
def test_profile_quality_gate_requires_all_matched_baselines(tmp_path: Path) -> None:
    """公共 profile 门禁必须拒绝缺少任一 baseline 配对质量的 decision。"""

    from evaluation.protocol.paper_profile_evidence_closure import (
        build_paper_profile_evidence_closure_audit,
    )
    from evaluation.protocol.record_writer import write_json

    run_root = tmp_path / "run"
    config_path = tmp_path / "quality_gate.json"
    protocol = _context()["video_quality_comparison_protocol"]
    config_path.write_text(json.dumps({
        "paper_result_level": "probe_paper",
        "target_fpr": 0.1,
        "require_baseline_matched_video_quality_metrics": True,
        "video_quality_comparison_protocol": protocol,
    }), encoding="utf-8")
    decision_path = run_root / "artifacts" / "video_quality_metric_decision.json"
    write_json(decision_path, {
        "target_fpr": 0.1,
        "video_quality_comparison_protocol": protocol,
        "sstw_paired_video_quality_ready": True,
        "baseline_matched_video_quality_ready": True,
        "video_quality_missing_method_ids": [],
    })

    passed = build_paper_profile_evidence_closure_audit(run_root, config_path)
    assert passed["paper_profile_evidence_closure_decision"] == "PASS"
    assert passed["paper_profile_evidence_closure_checks"][
        "baseline_matched_video_quality_passed"
    ] is True

    write_json(decision_path, {
        "target_fpr": 0.1,
        "video_quality_comparison_protocol": protocol,
        "sstw_paired_video_quality_ready": True,
        "baseline_matched_video_quality_ready": False,
        "video_quality_missing_method_ids": ["videoseal"],
    })
    failed = build_paper_profile_evidence_closure_audit(run_root, config_path)
    assert failed["paper_profile_evidence_closure_decision"] == "FAIL"
    assert failed["paper_profile_evidence_closure_missing_requirements"] == [
        "baseline_matched_video_quality_passed"
    ]

@pytest.mark.quick
def test_postprocess_reuses_governed_quality_records_without_restoring_baseline_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式比较阶段完成质量计算后, 论文后处理不得再次恢复 baseline 大视频包。"""

    run_root = tmp_path / "run"
    _write_quality_sources(run_root)
    monkeypatch.setattr(builders, "compute_paired_video_quality_metrics", lambda *_: {
        "paired_video_quality_status": "ready",
        "paired_watermark_psnr": 40.0,
        "paired_watermark_ssim": 0.98,
        "paired_temporal_delta_error": 0.01,
    })
    first = builders.run_video_quality_metric_artifact_builder(
        run_root,
        "configs/protocol/probe_paper_generative_probe.json",
    )
    assert first["video_quality_metric_decision"] == "PASS"
    for source in (run_root / "baseline_sources").glob("*.mp4"):
        source.unlink()

    monkeypatch.setattr(
        builders,
        "compute_paired_video_quality_metrics",
        lambda *_: (_ for _ in ()).throw(AssertionError("不应重新解码 baseline 视频")),
    )
    second = builders.run_video_quality_metric_artifact_builder(
        run_root,
        "configs/protocol/probe_paper_generative_probe.json",
        reuse_ready_records=True,
    )

    assert second["video_quality_metric_decision"] == "PASS"
    assert second["baseline_matched_video_quality_ready"] is True

@pytest.mark.quick
def test_quality_and_replay_calibration_artifacts_cross_stage_packages() -> None:
    """质量 records 与 replay 冻结校准必须随其生产阶段包进入后续会话。"""

    from workflows.stage_package_sync import (
        FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS,
        RUNTIME_DETECTION_PACKAGE_RELPATHS,
    )

    assert "records/video_quality_metric_records.jsonl" in FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS
    assert "artifacts/video_quality_metric_decision.json" in FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS
    assert "figures/video_quality_robustness_tradeoff_figure.json" in FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS
    assert (
        "thresholds/replay_gaussian_likelihood_calibrations.jsonl"
        in RUNTIME_DETECTION_PACKAGE_RELPATHS
    )
