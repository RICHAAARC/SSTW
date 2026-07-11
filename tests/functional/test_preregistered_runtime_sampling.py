from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from experiments.generative_video_model_probe.attack_runner import (
    _select_preregistered_attack_jobs,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _generation_causal_ablation_replay_sources,
)
from experiments.generative_video_model_probe.formal_method_variants import (
    GENERATION_METHOD_VARIANTS,
)


def _generation_record(
    index: int,
    *,
    method_variant: str = "sstw_full_method",
    split: str = "test",
) -> dict[str, object]:
    """构造只含预注册选择所需身份字段的轻量记录。"""

    return {
        "generation_status": "success",
        "generation_model_id": "model-a",
        "prompt_id": f"prompt-{index}",
        "seed_id": f"seed-{index}",
        "trajectory_trace_id": f"trace-{index}-{method_variant}",
        "split": split,
        "method_variant": method_variant,
        "sample_role": "attacked_positive_source",
        "video_path": f"video-{index}-{method_variant}.mp4",
        "video_sha256": f"digest-{index}-{method_variant}",
    }


@pytest.mark.quick
def test_runtime_attack_subset_is_full_method_bounded_and_score_independent() -> None:
    """主攻击子集必须在检测前按身份冻结，且不得混入内部消融。"""

    records = [
        _generation_record(index)
        for index in range(8)
    ] + [
        _generation_record(index, method_variant="without_velocity_constraint")
        for index in range(8)
    ]
    jobs = _select_preregistered_attack_jobs(
        records,
        ("attack-a", "attack-b"),
        maximum_per_model_split=3,
    )
    contaminated = [
        {**record, "S_final_conservative": 1.0 - index * 0.1}
        for index, record in enumerate(records)
    ]
    repeated = _select_preregistered_attack_jobs(
        contaminated,
        ("attack-a", "attack-b"),
        maximum_per_model_split=3,
    )

    assert Counter(attack for _record, attack, _rank, _digest in jobs) == {
        "attack-a": 3,
        "attack-b": 3,
    }
    assert all(record["method_variant"] == "sstw_full_method" for record, *_ in jobs)
    assert [record["trajectory_trace_id"] for record, *_ in jobs] == [
        record["trajectory_trace_id"] for record, *_ in repeated
    ]
    assert all(rank in {1, 2, 3} for _record, _attack, rank, _digest in jobs)


@pytest.mark.quick
def test_causal_ablation_subset_selects_complete_prompt_seed_blocks() -> None:
    """Claim-1 子集必须对每个身份同时保留全部生成级变体。"""

    records = [
        _generation_record(index, method_variant=variant)
        for index in range(6)
        for variant in GENERATION_METHOD_VARIANTS
    ]
    # 破坏一个身份的完整性，验证该身份不会形成不公平配对。
    records = [
        record
        for record in records
        if not (
            record["prompt_id"] == "prompt-5"
            and record["method_variant"] == "without_velocity_constraint"
        )
    ]
    selected = _generation_causal_ablation_replay_sources(
        records,
        maximum_identity_count_per_model_split=2,
    )
    variants_by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in selected:
        variants_by_identity[(str(record["prompt_id"]), str(record["seed_id"]))].add(
            str(record["method_variant"])
        )
        assert record["attack_name"] == "no_attack_generation_causal_ablation"
        assert record["attacked_video_path"] == record["video_path"]

    assert len(variants_by_identity) == 2
    assert all(variants == set(GENERATION_METHOD_VARIANTS) for variants in variants_by_identity.values())
    assert ("prompt-5", "seed-5") not in variants_by_identity
