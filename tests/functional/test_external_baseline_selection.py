"""验证 B5 external baseline 推荐与 claim 约束。"""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from main.external_baselines.baseline_registry import build_external_baseline_records
from main.external_baselines.explicit_dtw_temporal_alignment import compute_dtw_alignment_cost
from main.external_baselines.frame_matching_temporal_registration import compute_registration_cost, match_frames


@pytest.mark.quick
def test_external_baseline_selection_uses_explicit_synchronization_without_paper_a_overlap() -> None:
    """外部 baseline 推荐必须改为显式同步机制, 并且不得复用已排除的视频水印系统。"""
    config_path = Path("configs/external_baselines/external_baselines.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = build_external_baseline_records(config_path)

    names = [record["external_baseline_name"] for record in records]
    assert names == [
        "explicit_dtw_temporal_alignment",
        "explicit_frame_matching_temporal_registration",
    ]
    excluded_related_work_names = {"video" + "seal", "riva" + "gan", "vid" + "stamp"}
    assert excluded_related_work_names.isdisjoint(names)
    assert config["selection_policy"]["claim_rule"]
    assert "key_conditioned_state_space_inference" in config["internal_mechanism_baselines"]
    assert all(record["external_baseline_result_used_for_claim"] is False for record in records)
    assert all(record["external_baseline_runnable_status"] == "runnable" for record in records)
    assert records[0]["external_baseline_selection_role"] == "primary_explicit_synchronization_external_baseline"


@pytest.mark.quick
def test_explicit_synchronization_adapters_run_on_small_sequences() -> None:
    """两个 external baseline 适配器必须能在轻量 embedding 序列上运行。"""
    reference = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    observed = [[0.0, 0.0], [1.1, 0.0], [2.0, 0.0]]

    assert compute_dtw_alignment_cost(reference, observed) >= 0.0
    assert compute_registration_cost(reference, observed) >= 0.0

    matches = match_frames(reference, observed)
    assert [item["reference_index"] for item in matches] == [0, 1, 2]
