from pathlib import Path

import pytest

from evaluation.protocol.paper_result_formality_guard import build_paper_result_formality_guard
from evaluation.protocol.record_writer import write_json, write_jsonl


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


@pytest.mark.quick
def test_paper_result_formality_guard_scans_reports_figures_and_manifests(tmp_path: Path) -> None:
    """正式性门禁必须覆盖最终结果包中的 reports、figures 和 manifests。"""

    run_root = tmp_path / "run"
    (run_root / "reports").mkdir(parents=True)
    (run_root / "reports" / "paper_report.md").write_text(
        "# report\n\nthis line contains placeholder evidence\n",
        encoding="utf-8",
    )
    write_json(run_root / "figures" / "summary_figure.json", {
        "figure_id": "summary",
        "figure_rows": [{"metric_status": "measured_formal"}],
    })
    write_json(run_root / "manifests" / "package_manifest.json", {
        "manifest_kind": "paper_package_manifest",
        "claim_support_status": "ready",
    })

    audit = build_paper_result_formality_guard(
        run_root,
        paper_result_level="probe_paper",
        target_fpr=0.1,
    )

    assert audit["paper_result_formality_guard_decision"] == "FAIL"
    assert "placeholder" in audit["paper_result_formality_guard_blocking_terms"]
    assert any(
        violation["relative_path"] == "reports/paper_report.md"
        for violation in audit["paper_result_formality_guard_violations"]
    )


@pytest.mark.quick
def test_paper_result_formality_guard_allows_governance_prohibited_terms(tmp_path: Path) -> None:
    """禁止来源列表可以登记被拒绝的弱证据名称, 但不能作为结果值出现。"""

    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "external_baseline_execution_manifest.json", {
        "external_baseline_prohibited_result_sources": ["sstw_proxy_score"],
        "fail_closed_policy": "manual_result_json and sstw_proxy_score are rejected",
        "fallback_rule": "fallback entries are documented as blocked policy only",
        "metric_status": "measured_formal",
    })

    audit = build_paper_result_formality_guard(
        run_root,
        paper_result_level="probe_paper",
        target_fpr=0.1,
    )

    assert audit["paper_result_formality_guard_decision"] == "PASS"


@pytest.mark.quick
def test_paper_result_formality_guard_scans_external_baseline_status_records(tmp_path: Path) -> None:
    """external baseline 状态记录也属于结果包扫描范围, 不能藏入弱证据结果。"""

    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", [
        {
            "external_baseline_name": "bad_baseline",
            "metric_status": "measured_formal",
            "external_baseline_score_source": "manual_proxy_result",
        }
    ])

    audit = build_paper_result_formality_guard(
        run_root,
        paper_result_level="probe_paper",
        target_fpr=0.1,
    )

    assert audit["paper_result_formality_guard_decision"] == "FAIL"
    assert "proxy" in audit["paper_result_formality_guard_blocking_terms"]
