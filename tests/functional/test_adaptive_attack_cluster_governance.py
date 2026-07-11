"""验证 adaptive attack 的独立簇覆盖与无伪重复契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.generative_video_model_probe.formal_adaptive_attack_executor import (
    _generation_spoof_source_population,
    _select_preregistered_source_clusters,
    audit_adaptive_source_cluster_coverage,
)
from evaluation.protocol.record_writer import write_jsonl


@pytest.mark.quick
def test_source_cluster_selection_is_score_independent_and_deterministic() -> None:
    """预注册子集选择不得随 detector 分数或输入行顺序改变。"""

    rows = [
        {
            "statistical_cluster_id": f"cluster-{index}",
            "detector_score": float(index),
        }
        for index in range(8)
    ]
    selected = _select_preregistered_source_clusters(
        rows,
        selected_cluster_count=4,
    )
    score_changed = _select_preregistered_source_clusters(
        [
            {**row, "detector_score": 100.0 - row["detector_score"]}
            for row in reversed(rows)
        ],
        selected_cluster_count=4,
    )

    assert {
        row["statistical_cluster_id"] for row in selected
    } == {
        row["statistical_cluster_id"] for row in score_changed
    }


@pytest.mark.quick
def test_cluster_audit_uses_one_source_unit_and_one_disjoint_collusion_pair() -> None:
    """重复查询不是独立样本, collusion 每个不重叠视频对只能出现一次。"""

    clusters = [f"cluster-{index}" for index in range(4)]
    protocol = "watermark_removal_optimization_attack"
    records = [
        {
            "non_runtime_attack_protocol": protocol,
            "statistical_cluster_id": cluster,
            "adaptive_attack_source_statistical_cluster_id": cluster,
        }
        for cluster in clusters
    ]
    records.extend([
        {
            "non_runtime_attack_protocol": "collusion_multi_sample_attack",
            "statistical_cluster_id": "pair-0",
            "statistical_independent_unit": "disjoint_source_video_pair",
            "adaptive_attack_source_statistical_cluster_id": clusters[0],
            "adaptive_attack_member_statistical_cluster_ids": clusters[:2],
        },
        {
            "non_runtime_attack_protocol": "collusion_multi_sample_attack",
            "statistical_cluster_id": "pair-1",
            "statistical_independent_unit": "disjoint_source_video_pair",
            "adaptive_attack_source_statistical_cluster_id": clusters[2],
            "adaptive_attack_member_statistical_cluster_ids": clusters[2:],
        },
    ])
    checkpoints = [
        {
            "non_runtime_attack_protocol": protocol,
            "statistical_cluster_id": cluster,
            "adaptive_attack_query_budget_checkpoint": budget,
            "adaptive_attack_checkpoint_has_admissible_candidate": True,
            "adaptive_attack_checkpoint_output_video_sha256": f"{cluster}-{budget}",
        }
        for budget in (1, 9)
        for cluster in clusters
    ]

    ready = audit_adaptive_source_cluster_coverage(
        records,
        checkpoints,
        required_protocols=(protocol, "collusion_multi_sample_attack"),
        minimum_source_cluster_count=4,
        query_budget_checkpoints=(1, 9),
    )
    duplicated = audit_adaptive_source_cluster_coverage(
        [*records, records[-1]],
        checkpoints,
        required_protocols=(protocol, "collusion_multi_sample_attack"),
        minimum_source_cluster_count=4,
        query_budget_checkpoints=(1, 9),
    )

    assert ready["adaptive_attack_source_cluster_coverage_decision"] == "PASS"
    assert ready["adaptive_attack_independent_unit_uniqueness_decision"] == "PASS"
    assert ready[
        "adaptive_attack_query_budget_checkpoint_coverage_decision"
    ] == "PASS"
    assert duplicated[
        "adaptive_attack_independent_unit_uniqueness_decision"
    ] == "FAIL"


@pytest.mark.quick
def test_spoof_fixed_fpr_sample_can_exceed_costly_adaptive_search_sample() -> None:
    """低 FPR spoof 统计可扩大廉价单查询样本, 不得伪造 adaptive query 数。"""

    retention_clusters = {f"cluster-{index}" for index in range(4)}
    spoof_clusters = {f"cluster-{index}" for index in range(6)}
    retention_protocol = "watermark_removal_optimization_attack"
    spoof_protocol = "watermark_spoofing_or_copy_attack"
    records = [
        {
            "non_runtime_attack_protocol": retention_protocol,
            "statistical_cluster_id": cluster,
            "adaptive_attack_source_statistical_cluster_id": cluster,
        }
        for cluster in retention_clusters
    ] + [
        {
            "non_runtime_attack_protocol": spoof_protocol,
            "statistical_cluster_id": cluster,
            "adaptive_attack_source_statistical_cluster_id": cluster,
        }
        for cluster in spoof_clusters
    ]
    checkpoints = [
        {
            "non_runtime_attack_protocol": retention_protocol,
            "statistical_cluster_id": cluster,
            "adaptive_attack_query_budget_checkpoint": 9,
            "adaptive_attack_checkpoint_has_admissible_candidate": True,
            "adaptive_attack_checkpoint_output_video_sha256": f"digest-{cluster}",
        }
        for cluster in retention_clusters
    ]

    audit = audit_adaptive_source_cluster_coverage(
        records,
        checkpoints,
        required_protocols=(retention_protocol, spoof_protocol),
        minimum_source_cluster_count=4,
        minimum_spoof_source_cluster_count=6,
        query_budget_checkpoints=(9,),
        expected_retention_source_cluster_ids=retention_clusters,
        expected_spoof_source_cluster_ids=spoof_clusters,
    )

    assert audit["adaptive_attack_source_cluster_coverage_decision"] == "PASS"
    assert audit["adaptive_attack_selected_source_video_cluster_count"] == 4
    assert audit["adaptive_attack_spoof_source_video_cluster_count"] == 6


@pytest.mark.quick
def test_spoof_population_uses_all_real_heldout_generation_donors(
    tmp_path: Path,
) -> None:
    """copy/spoof donor 总体不得受逐 runtime attack 的小子集上限约束。"""

    generation_records = []
    clean_rows = []
    for index in range(6):
        identity = {
            "generation_model_id": "wan-small",
            "prompt_id": f"prompt-{index}",
            "seed_id": "seed-0",
        }
        generation_records.append({
            **identity,
            "generation_status": "success",
            "sample_role": "attacked_positive_source",
            "method_variant": "sstw_full_method",
            "split": "test",
            "cross_model_role": "primary_generation_model",
            "video_path": f"video-{index}.mp4",
            "video_sha256": f"digest-{index}",
        })
        clean_rows.append({
            **identity,
            "statistical_cluster_id": f"cluster-{index}",
            "attack_name": "clean_negative",
        })
    write_jsonl(
        tmp_path / "records" / "generation_records.jsonl",
        generation_records,
    )

    population = _generation_spoof_source_population(tmp_path, clean_rows)

    assert len(population) == 6
    assert {
        row["statistical_cluster_id"] for row in population
    } == {f"cluster-{index}" for index in range(6)}
    assert all(
        row["adaptive_attack_spoof_donor_source"]
        == "heldout_full_method_generation_video"
        for row in population
    )
