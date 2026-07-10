"""验证三层论文主张的审稿证据索引."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.reviewer_evidence_index import (
    CLAIM_EVIDENCE_PATHS,
    build_reviewer_evidence_index,
    write_reviewer_evidence_index,
)


def _write_text(path: Path, content: str = "evidence\n") -> None:
    """写入最小真实文件, 供索引器计算内容摘要."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_complete_claim_evidence(run_root: Path) -> None:
    """构造三个 claim 均通过且全部 artifact 存在的最小 fixture."""

    for relative_paths in CLAIM_EVIDENCE_PATHS.values():
        for relative_path in relative_paths:
            _write_text(run_root / relative_path)

    gate_path = run_root / "artifacts" / "probe_paper_gate_decision.json"
    gate_path.write_text(json.dumps({
        "probe_paper_gate_decision": "PASS",
        "paper_result_level": "probe_paper",
    }), encoding="utf-8")
    complete_path = run_root / "artifacts" / "complete_paper_mechanism_claim_decision.json"
    complete_path.write_text(json.dumps({
        "complete_paper_mechanism_claim_decision": "PASS",
        "claim_1_velocity_constraint_detectable_watermark_decision": "PASS",
        "claim_2_path_evidence_independent_gain_decision": "PASS",
        "claim_3_attacked_video_replay_posterior_decision": "PASS",
    }), encoding="utf-8")


@pytest.mark.quick
def test_reviewer_index_requires_passed_gate_and_real_artifacts(tmp_path: Path) -> None:
    """索引 PASS 必须同时依赖 profile gate、三层 claim 和真实文件."""

    run_root = tmp_path / "run"
    _seed_complete_claim_evidence(run_root)

    index = write_reviewer_evidence_index(run_root)

    assert index["reviewer_evidence_index_decision"] == "PASS"
    assert index["indexed_claim_count"] == 3
    assert index["missing_evidence_paths"] == []
    assert all(row["evidence_sha256"] for row in index["evidence_rows"])
    assert (run_root / "reports" / "reviewer_evidence_index.md").is_file()


@pytest.mark.quick
def test_reviewer_index_fails_closed_when_evidence_is_missing(tmp_path: Path) -> None:
    """删除任一必需 artifact 后必须 FAIL, 不能保留 supported claim."""

    run_root = tmp_path / "run"
    _seed_complete_claim_evidence(run_root)
    missing_path = run_root / "records" / "wrong_time_grid_replay_records.jsonl"
    missing_path.unlink()

    index = build_reviewer_evidence_index(run_root)

    assert index["reviewer_evidence_index_decision"] == "FAIL"
    assert "records/wrong_time_grid_replay_records.jsonl" in index["missing_evidence_paths"]
    assert index["claim_support_status"] == "probe_paper_reviewer_evidence_index_blocked"
