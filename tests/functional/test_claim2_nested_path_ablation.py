"""验证 Claim-2 嵌套路径消融与 replay 路径权重的真实语义。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _audit_three_layer_mechanism,
    _paired_path_gain_records,
    _score_records_with_frozen_calibration,
)
from experiments.generative_video_model_probe.formal_method_variants import (
    CLAIM2_PATH_NESTED_ABLATION_VARIANT,
    DETECTOR_ONLY_METHOD_VARIANTS,
    FORMAL_DETECTOR_VARIANTS,
    GENERATION_METHOD_VARIANTS,
    transform_flow_evidence_record_for_method_variant,
)
from main.methods.state_space_watermark.path_observation import (
    aggregate_path_observations,
)


def _phase(phase: float) -> dict[str, float]:
    """构造包含全部 Claim-2 相关特征的最小 phase 观测。"""

    return {
        "flow_phase": phase,
        "endpoint_score": 0.7,
        "velocity_score": 0.6,
        "path_score": 0.45,
        "path_score_unweighted": 0.5,
        "path_endpoint_consistency": 0.36,
        "path_endpoint_consistency_unweighted": 0.4,
        "replay_log_likelihood_ratio": 0.2,
        "replay_reliability": 0.8,
        "replay_reliability_weight": 0.8,
        "time_grid_reliability": 0.9,
        "coverage_ratio": 1.0,
    }


@pytest.mark.quick
def test_claim2_nested_ablation_only_removes_path_features() -> None:
    """Claim-2 消融不得同时移除 endpoint、velocity、replay 或 SSM 输入。"""

    record = {"flow_state_observation_sequence": [_phase(0.2), _phase(0.8)]}
    transformed = transform_flow_evidence_record_for_method_variant(
        record,
        method_variant=CLAIM2_PATH_NESTED_ABLATION_VARIANT,
    )

    for original, ablated in zip(
        record["flow_state_observation_sequence"],
        transformed["flow_state_observation_sequence"],
    ):
        assert ablated["path_score"] == 0.0
        assert ablated["path_endpoint_consistency"] == 0.0
        for unchanged in (
            "flow_phase",
            "endpoint_score",
            "velocity_score",
            "replay_log_likelihood_ratio",
            "replay_reliability",
            "time_grid_reliability",
            "coverage_ratio",
        ):
            assert ablated[unchanged] == original[unchanged]


@dataclass(frozen=True)
class _Calibration:
    """为 calibration 来源隔离测试保存检测器身份。"""

    detector_configuration_id: str

    def as_dict(self) -> dict[str, object]:
        """返回 threshold row 所需的最小可序列化数据。"""

        return {"detector_configuration_id": self.detector_configuration_id}


@pytest.mark.quick
def test_claim2_detector_calibration_reuses_full_method_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检测器级路径消融必须复用 full-method calibration, 不要求重复生成视频。"""

    rows = [
        {
            "generation_model_id": "model-a",
            "method_variant": variant,
            "split": "calibration",
            "sample_role": "clean_negative",
            "formal_flow_evidence_unit_id": f"unit::{variant}",
        }
        for variant in GENERATION_METHOD_VARIANTS
    ]
    calibration_sources: dict[str, set[str]] = {}

    def fake_fit(records, *, method_variant, target_fpr):
        del target_fpr
        calibration_sources[method_variant] = {
            str(record["method_variant"]) for record in records
        }
        return _Calibration(method_variant)

    def fake_apply(record, calibration):
        del record
        return {
            "S_final_conservative": 0.2,
            "decision": False,
            "target_fpr": 0.1,
            "detector_configuration_id": calibration.detector_configuration_id,
        }

    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner."
        "fit_flow_evidence_calibration",
        fake_fit,
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner."
        "apply_frozen_flow_detector",
        fake_apply,
    )

    _scored, thresholds, calibrations = _score_records_with_frozen_calibration(
        rows,
        target_fpr=0.1,
    )

    assert set(calibrations) == {
        ("model-a", variant) for variant in FORMAL_DETECTOR_VARIANTS
    }
    assert calibration_sources[CLAIM2_PATH_NESTED_ABLATION_VARIANT] == {
        "sstw_full_method"
    }
    assert all(
        calibration_sources[variant] == {"sstw_full_method"}
        for variant in DETECTOR_ONLY_METHOD_VARIANTS
    )
    nested_threshold = next(
        row
        for row in thresholds
        if row["method_variant"] == CLAIM2_PATH_NESTED_ABLATION_VARIANT
    )
    assert nested_threshold["calibration_source_method_variant"] == "sstw_full_method"
    assert nested_threshold["detector_only_nested_ablation"] is True


@pytest.mark.quick
def test_formal_variant_partition_matches_governed_contract() -> None:
    """生成级与检测器级变体分区必须和完整机制契约完全一致。"""

    contract = json.loads(
        Path("configs/protocol/sstw_complete_paper_mechanism.json").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(contract["generation_internal_ablation_variants"]) == (
        *GENERATION_METHOD_VARIANTS,
    )
    assert tuple(contract["detector_only_internal_ablation_variants"]) == (
        *DETECTOR_ONLY_METHOD_VARIANTS,
    )
    assert set(GENERATION_METHOD_VARIANTS).isdisjoint(DETECTOR_ONLY_METHOD_VARIANTS)
    assert set(GENERATION_METHOD_VARIANTS) | set(DETECTOR_ONLY_METHOD_VARIANTS) == set(
        contract["formal_method_variants"]
    )


@pytest.mark.quick
def test_detector_only_source_and_duplicate_units_are_rejected() -> None:
    """正式评分不得消费旧检测器专用视频或重复 governed replay 单元。"""

    detector_only_source = {
        "generation_model_id": "model-a",
        "method_variant": "trajectory_only_score",
        "split": "calibration",
        "sample_role": "clean_negative",
        "formal_flow_evidence_unit_id": "unit-a",
    }
    with pytest.raises(ValueError, match="检测器专用消融"):
        _score_records_with_frozen_calibration(
            [detector_only_source],
            target_fpr=0.1,
        )

    duplicate_full = {
        **detector_only_source,
        "method_variant": "sstw_full_method",
    }
    with pytest.raises(RuntimeError, match="重复 formal_flow_evidence_unit_id"):
        _score_records_with_frozen_calibration(
            [duplicate_full, dict(duplicate_full)],
            target_fpr=0.1,
        )


@pytest.mark.quick
def test_claim2_pair_compares_full_with_without_path_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配对记录必须明确绑定仅移除路径证据的冻结检测器。"""

    @dataclass(frozen=True)
    class Calibration:
        detector_configuration_id: str

    def fake_apply(record, calibration):
        del record
        return {
            "S_final_conservative": 0.4,
            "decision": False,
            "target_fpr": 0.1,
            "detector_configuration_id": calibration.detector_configuration_id,
        }

    monkeypatch.setattr(
        "experiments.generative_video_model_probe.formal_flow_evidence_runner."
        "apply_frozen_flow_detector",
        fake_apply,
    )
    scored = [{
        "generation_model_id": "model-a",
        "sample_role": "attacked_positive",
        "method_variant": "sstw_full_method",
        "split": "test",
        "S_final_conservative": 0.7,
        "decision": True,
        "target_fpr": 0.1,
        "statistical_cluster_id": "video-a",
    }]
    calibrations = {
        ("model-a", CLAIM2_PATH_NESTED_ABLATION_VARIANT): Calibration(
            CLAIM2_PATH_NESTED_ABLATION_VARIANT
        )
    }

    [paired] = _paired_path_gain_records(scored, calibrations)

    assert paired["paired_path_ablation_method_variant"] == "without_path_evidence"
    assert paired["paired_without_path_evidence_detector_score"] == pytest.approx(0.4)
    assert paired["paired_path_evidence_score_gain"] == pytest.approx(0.3)
    assert paired["paired_path_evidence_detection_gain"] == 1
    assert paired["paired_fpr_alignment_status"] == "same_preregistered_target_fpr"
    assert paired["paired_path_nested_ablation_status"] == (
        "same_video_same_replay_only_path_features_removed"
    )


@pytest.mark.quick
def test_path_aggregation_directly_applies_replay_step_reliability() -> None:
    """低可靠 replay step 必须直接衰减路径积分, 而不是只写旁路字段。"""

    base_rows = [
        {
            "flow_phase": 0.25,
            "path_projection_normalized": 0.8,
            "velocity_projection_normalized": 0.3,
            "path_velocity_consistency": 0.9,
        },
        {
            "flow_phase": 0.75,
            "path_projection_normalized": 0.8,
            "velocity_projection_normalized": 0.3,
            "path_velocity_consistency": 0.9,
        },
    ]
    unweighted = aggregate_path_observations(base_rows)
    weighted = aggregate_path_observations([
        {**row, "replay_reliability_weight": 0.25} for row in base_rows
    ])

    assert unweighted["S_path_inv"] == pytest.approx(0.8)
    assert weighted["S_path_inv_unweighted"] == pytest.approx(0.8)
    assert weighted["S_path_inv"] == pytest.approx(0.2)
    assert weighted["path_replay_reliability_weight_mean"] == pytest.approx(0.25)
    assert weighted["path_replay_weighted_aggregation_applied"] is True


@pytest.mark.quick
def test_claim2_gate_rejects_non_nested_or_incomplete_pairs() -> None:
    """Claim-2 必须覆盖全部 held-out 视频并满足仅路径单一干预。"""

    positive_records = [
        {
            "sample_role": "attacked_positive",
            "method_variant": "sstw_full_method",
            "split": "test",
            "decision": True,
            "statistical_cluster_id": f"positive-{index}",
        }
        for index in range(12)
    ]
    negative_records = [
        {
            "sample_role": "clean_negative",
            "method_variant": "sstw_full_method",
            "split": "test",
            "decision": False,
            "statistical_cluster_id": f"negative-{index}",
        }
        for index in range(12)
    ]
    paired_path = [
        {
            "statistical_cluster_id": f"positive-{index}",
            "paired_path_ablation_method_variant": "without_path_evidence",
            "paired_path_nested_ablation_status": (
                "same_video_same_replay_only_path_features_removed"
            ),
            "paired_detector_threshold_source_split": "calibration",
            "paired_test_time_threshold_update_blocked": True,
            "paired_fpr_alignment_status": "same_preregistered_target_fpr",
            "target_fpr": 0.1,
            "paired_path_evidence_score_gain": 0.3,
            "paired_path_evidence_detection_gain": 1,
        }
        for index in range(12)
    ]
    paired_velocity = [
        {
            "statistical_cluster_id": f"positive-{index}",
            "velocity_causal_pairing_status": "matched_single_intervention_design",
            "metric_status": "measured_formal",
            "paired_velocity_causal_score_gain": 0.3,
            "paired_velocity_causal_detection_gain": 1,
        }
        for index in range(12)
    ]

    passed = _audit_three_layer_mechanism(
        positive_records + negative_records,
        paired_path,
        paired_velocity,
        target_fpr=0.1,
    )
    contaminated = [dict(row) for row in paired_path]
    contaminated[0]["paired_path_ablation_method_variant"] = "endpoint_only_control"
    failed = _audit_three_layer_mechanism(
        positive_records + negative_records,
        contaminated,
        paired_velocity,
        target_fpr=0.1,
    )

    assert passed["claim_2_path_evidence_independent_gain_decision"] == "PASS"
    assert passed["claim_2_paired_comparison_coverage"] == 1.0
    assert failed["claim_2_path_evidence_independent_gain_decision"] == "FAIL"
    assert failed["claim_2_pairing_failure_count"] == 1
