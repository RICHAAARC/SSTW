"""验证 external baseline 差值统计按生成设计分流。"""

from __future__ import annotations

from pathlib import Path

import pytest

import experiments.generative_video_model_probe.formal_baseline_difference_interval as intervals
from evaluation.protocol.record_writer import write_jsonl


pytestmark = pytest.mark.quick


def _detection_units(values: tuple[bool, bool, bool, bool]) -> list[dict[str, object]]:
    """构造2个独立 source-video 簇和2种 attack 的检测单元。"""

    anchors = (
        "prompt_0::seed_0::attack_a",
        "prompt_0::seed_0::attack_b",
        "prompt_1::seed_1::attack_a",
        "prompt_1::seed_1::attack_b",
    )
    return [
        {
            "comparison_anchor_key": anchor,
            "detected_at_target_fpr": detected,
        }
        for anchor, detected in zip(anchors, values)
    ]


def test_native_generation_uses_independent_interval_without_paired_legacy_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native-generation 只能共享设计块, 不能伪装成同一视频配对。"""

    run_root = tmp_path / "run"
    write_jsonl(
        run_root / "records" / "fair_detection_calibration_records.jsonl",
        [
            {
                "method_id": intervals.SSTW_METHOD_ID,
                "target_fpr": 0.1,
                "tpr_at_target_fpr": 0.75,
                "attacked_positive_score_count": 4,
                "positive_detection_units_at_target_fpr": _detection_units(
                    (True, True, True, False)
                ),
            },
            {
                "method_id": "videoshield",
                "target_fpr": 0.1,
                "tpr_at_target_fpr": 0.25,
                "attacked_positive_score_count": 4,
                "positive_detection_units_at_target_fpr": _detection_units(
                    (False, True, False, False)
                ),
            },
            {
                "method_id": "videoseal",
                "target_fpr": 0.1,
                "tpr_at_target_fpr": 0.25,
                "attacked_positive_score_count": 4,
                "positive_detection_units_at_target_fpr": _detection_units(
                    (False, True, False, False)
                ),
            },
        ],
    )
    monkeypatch.setattr(
        intervals,
        "_load_profile_context",
        lambda _path: {
            "paper_result_level": "probe_paper",
            "target_fpr": 0.1,
            "target_fpr_source_config_path": "test_config.json",
            "required_modern_external_baseline_adapter_names": [
                "videoshield",
                "videoseal",
            ],
            "required_runtime_attack_names": ["attack_a", "attack_b"],
            "allow_effect_size_claims": True,
        },
    )

    rows = intervals.build_formal_baseline_difference_interval_records(
        run_root,
        "test_config.json",
    )
    by_method = {row["baseline_method_id"]: row for row in rows}
    native = by_method["videoshield"]
    posthoc = by_method["videoseal"]

    assert native["difference_interval_status"] == "ready"
    assert native["difference_interval_method"] == (
        "independent_two_sample_source_video_cluster_bootstrap_detection_difference"
    )
    assert native["paired_source_video_inference_used"] is False
    assert native["source_video_pairing_status"] == (
        "independent_native_generation_videos_not_cross_generator_paired"
    )
    assert native["paired_comparison_unit_count"] == 0
    assert native["paired_comparison_anchor_keys"] == []
    assert native["paired_attack_names"] == []
    assert native["matched_design_anchor_count"] == 4
    assert len(native["matched_design_anchor_keys"]) == 4
    assert native["matched_design_attack_names"] == ["attack_a", "attack_b"]

    assert posthoc["difference_interval_status"] == "ready"
    assert posthoc["difference_interval_method"] == (
        "paired_source_video_cluster_bootstrap_detection_difference"
    )
    assert posthoc["paired_source_video_inference_used"] is True
    assert posthoc["paired_comparison_unit_count"] == 4
    assert len(posthoc["paired_comparison_anchor_keys"]) == 4
