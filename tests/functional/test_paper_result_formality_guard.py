from pathlib import Path

import pytest

from main.protocol.paper_result_formality_guard import build_paper_result_formality_guard
from main.protocol.record_writer import write_json, write_jsonl


@pytest.mark.quick
def test_paper_result_formality_guard_allows_clean_formal_records(tmp_path: Path) -> None:
    """正式结果包只包含 measured_formal 证据时, 允许 probe_claim。"""

    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "runtime_attack_records.jsonl", [
        {
            "attack_name": "video_compression_runtime",
            "metric_status": "measured_formal",
            "runtime_attack_proxy_free": True,
        }
    ])
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS",
        "target_fpr": 0.1,
    })

    audit = build_paper_result_formality_guard(
        run_root,
        paper_result_level="probe_paper",
        target_fpr=0.1,
    )

    assert audit["paper_result_formality_guard_decision"] == "PASS"
    assert audit["paper_claim_id"] == "probe_claim"
    assert audit["paper_claim_support_status"] == "probe_claim_supported"
    assert audit["paper_result_formality_guard_violation_count"] == 0


@pytest.mark.quick
def test_paper_result_formality_guard_rejects_proxy_records(tmp_path: Path) -> None:
    """三层 paper 结果包中出现 proxy 结果时必须 fail-closed。"""

    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", [
        {
            "external_baseline_name": "explicit_control",
            "metric_status": "measured_proxy",
            "claim_support_status": "external_baseline_proxy_comparison_not_claim_supporting",
        }
    ])

    audit = build_paper_result_formality_guard(
        run_root,
        paper_result_level="pilot_paper",
        target_fpr=0.01,
    )

    assert audit["paper_result_formality_guard_decision"] == "FAIL"
    assert audit["paper_claim_id"] == "pilot_claim"
    assert audit["paper_claim_support_status"] == "pilot_claim_blocked"
    assert "proxy" in audit["paper_result_formality_guard_blocking_terms"]


@pytest.mark.quick
def test_paper_result_formality_guard_rejects_placeholder_fields(tmp_path: Path) -> None:
    """placeholder 字段不能进入正式 paper 结果包。"""

    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "metric_status": "measured_formal",
            "sampler_signature_placeholder": "pending",
        }
    ])

    audit = build_paper_result_formality_guard(
        run_root,
        paper_result_level="full_paper",
        target_fpr=0.001,
    )

    assert audit["paper_result_formality_guard_decision"] == "FAIL"
    assert audit["paper_claim_id"] == "full_claim"
    assert audit["paper_claim_support_status"] == "full_claim_blocked"
    assert "placeholder" in audit["paper_result_formality_guard_blocking_terms"]


@pytest.mark.quick
def test_paper_result_formality_guard_rejects_fallback_values(tmp_path: Path) -> None:
    """fallback 不能作为三层 paper 正式结果包的一部分。"""

    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", {
        "runtime_detection_decision": "PASS",
        "detection_mode": "fallback_detector",
    })

    audit = build_paper_result_formality_guard(
        run_root,
        paper_result_level="probe_paper",
        target_fpr=0.1,
    )

    assert audit["paper_result_formality_guard_decision"] == "FAIL"
    assert "fallback" in audit["paper_result_formality_guard_blocking_terms"]

