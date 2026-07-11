"""验证 Claim-3 held-out 概率可靠性统计不会复用 calibration 结果。"""

from __future__ import annotations

import pytest

from evaluation.statistics.probability_calibration import (
    cluster_balanced_probability_metrics,
    clustered_probability_calibration_interval,
)
from experiments.generative_video_model_probe.heldout_posterior_calibration import (
    audit_heldout_posterior_calibration_records,
    build_heldout_posterior_calibration_records,
)


def _config() -> dict:
    return {
        "required_runtime_attack_names": ["codec_attack", "frame_drop_attack"],
        "maximum_heldout_posterior_brier_score": 0.25,
        "maximum_heldout_posterior_log_loss": 0.693147,
        "maximum_heldout_posterior_expected_calibration_error": 0.1,
        "minimum_heldout_posterior_positive_cluster_count": 2,
        "minimum_heldout_posterior_negative_cluster_count": 2,
        "minimum_heldout_posterior_attack_cluster_count": 2,
        "heldout_posterior_bootstrap_resample_count": 100,
    }


def _record(
    record_id: str,
    *,
    sample_role: str,
    cluster_id: str,
    probability: float,
    attack_name: str | None,
) -> dict:
    return {
        "formal_flow_evidence_unit_id": record_id,
        "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "cross_model_role": "primary_claim_model",
        "method_variant": "sstw_full_method",
        "sample_role": sample_role,
        "attack_name": attack_name,
        "split": "test",
        "statistical_cluster_id": cluster_id,
        "flow_watermark_posterior_probability": probability,
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
        "metric_status": "measured_formal",
    }


def _heldout_rows() -> list[dict]:
    rows = [
        _record(
            "negative-1",
            sample_role="clean_negative",
            cluster_id="negative-cluster-1",
            probability=0.02,
            attack_name=None,
        ),
        _record(
            "negative-2",
            sample_role="clean_negative",
            cluster_id="negative-cluster-2",
            probability=0.04,
            attack_name=None,
        ),
    ]
    for attack_name in ("codec_attack", "frame_drop_attack"):
        rows.extend([
            _record(
                f"{attack_name}-positive-1",
                sample_role="attacked_positive",
                cluster_id="positive-cluster-1",
                probability=0.96,
                attack_name=attack_name,
            ),
            _record(
                f"{attack_name}-positive-2",
                sample_role="attacked_positive",
                cluster_id="positive-cluster-2",
                probability=0.94,
                attack_name=attack_name,
            ),
        ])
    return rows


@pytest.mark.quick
def test_cluster_balanced_probability_metrics_do_not_count_repeated_trials_as_clusters() -> None:
    probabilities = [0.02, 0.02, 0.96, 0.96]
    labels = [0, 0, 1, 1]
    cluster_ids = ["negative", "negative", "positive", "positive"]

    brier, log_loss, ece = cluster_balanced_probability_metrics(
        probabilities,
        labels,
        cluster_ids,
    )
    estimate = clustered_probability_calibration_interval(
        probabilities,
        labels,
        cluster_ids,
        bootstrap_resample_count=20,
    )

    assert brier < 0.01
    assert log_loss < 0.1
    assert ece < 0.1
    assert estimate.positive_cluster_count == 1
    assert estimate.negative_cluster_count == 1


@pytest.mark.quick
def test_heldout_posterior_requires_global_and_every_preregistered_attack() -> None:
    config = _config()
    records = build_heldout_posterior_calibration_records(
        _heldout_rows(),
        config,
    )
    audit = audit_heldout_posterior_calibration_records(records, config)

    assert len(records) == 3
    assert all(record["split"] == "test" for record in records)
    assert all(record["threshold_source_split"] == "calibration" for record in records)
    assert all(record["heldout_posterior_calibration_ready"] for record in records)
    assert audit["heldout_posterior_calibration_decision"] == "PASS"


@pytest.mark.quick
def test_heldout_posterior_blocks_bad_reliability_and_missing_attack_scope() -> None:
    config = _config()
    rows = _heldout_rows()
    for row in rows:
        if row["sample_role"] == "attacked_positive":
            row["flow_watermark_posterior_probability"] = 0.1
    records = build_heldout_posterior_calibration_records(rows, config)
    records = [
        record for record in records
        if record.get("attack_name") != "frame_drop_attack"
    ]
    audit = audit_heldout_posterior_calibration_records(records, config)

    assert audit["heldout_posterior_calibration_decision"] == "FAIL"
    assert audit["heldout_posterior_blocked_scopes"]
    assert any(
        "frame_drop_attack" in scope
        for scope in audit["heldout_posterior_missing_scopes"]
    )


@pytest.mark.quick
def test_heldout_posterior_rejects_calibration_rows() -> None:
    config = _config()
    rows = _heldout_rows()
    rows[0]["split"] = "calibration"

    # calibration 行不会进入 held-out 统计，剩余 negative 簇不足并由正式门禁阻断。
    records = build_heldout_posterior_calibration_records(rows, config)
    assert any(
        "heldout_negative_cluster_count_below_minimum"
        in record["heldout_posterior_calibration_failure_reasons"]
        for record in records
    )
