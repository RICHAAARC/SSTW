"""验证 full_paper result checker 的轻量工程门禁。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.protocol.record_writer import write_json, write_jsonl
from scripts.check_results.full_paper_result_checker import build_full_paper_result_checker_audit


def _write_pass_decision(path: Path, decision_field: str, target_fpr: float | None = None) -> None:
    """写入 checker 所需的轻量 PASS decision fixture。"""

    payload = {decision_field: "PASS"}
    if target_fpr is not None:
        payload["target_fpr"] = target_fpr
    if decision_field == "low_fpr_formal_statistics_decision":
        payload["current_profile_low_fpr_claim_allowed"] = True
    write_json(path, payload)


@pytest.mark.quick
def test_full_paper_result_checker_validates_split_and_per_attack_coverage(tmp_path: Path) -> None:
    """full_paper checker 必须检查 split、样本网格和每个 attack 的事件覆盖。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "full_paper_config.json"
    target_fpr = 0.001
    config = {
        "paper_result_level": "full_paper",
        "target_fpr": target_fpr,
        "minimum_prompt_count": 2,
        "minimum_seed_per_prompt": 2,
        "minimum_calibration_seed_per_prompt": 1,
        "minimum_test_seed_per_prompt": 1,
        "minimum_unique_video_count": 4,
        "minimum_calibration_unique_video_count": 2,
        "minimum_test_unique_video_count": 2,
        "minimum_attack_event_count_per_attack": 4,
        "minimum_heldout_attacked_positive_event_count": 8,
        "minimum_clean_negative_count": 4,
        "minimum_heldout_test_negative_event_count": 4,
        "required_runtime_attack_names": ["attack_a", "attack_b"],
        "required_modern_external_baseline_adapter_names": ["baseline_a"],
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    generation_records = []
    for prompt_index in range(2):
        for seed_index, split in enumerate(("calibration", "test")):
            generation_records.append({
                "generation_status": "success",
                "colab_runtime_profile": "full_paper",
                "generation_model_id": "model",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
                "trajectory_trace_id": f"trace_{prompt_index}_{seed_index}",
                "split": split,
            })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)

    attack_records = []
    detection_records = []
    for record in generation_records:
        for attack_name in ("attack_a", "attack_b"):
            base = {
                "generation_model_id": record["generation_model_id"],
                "prompt_id": record["prompt_id"],
                "seed_id": record["seed_id"],
                "trajectory_trace_id": record["trajectory_trace_id"],
                "attack_name": attack_name,
            }
            attack_records.append({**base, "attack_runtime_status": "ready"})
            detection_records.append({**base, "runtime_detection_status": "ready"})
    write_jsonl(run_root / "records" / "runtime_attack_records.jsonl", attack_records)
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", detection_records)

    write_json(run_root / "artifacts" / "runtime_attack_decision.json", {
        "runtime_attack_decision": "PASS",
        "runtime_attack_ready_count": len(attack_records),
        "runtime_attack_missing_required_names": [],
    })
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", {
        "runtime_detection_decision": "PASS",
        "runtime_detection_ready_count": len(detection_records),
        "runtime_detection_missing_required_names": [],
    })
    write_json(run_root / "artifacts" / "pilot_paper_to_full_paper_transition_decision.json", {
        "pilot_paper_to_full_paper_transition_decision": "PASS",
    })
    _write_pass_decision(run_root / "artifacts" / "fair_detection_calibration_decision.json", "fair_detection_calibration_decision", target_fpr)
    _write_pass_decision(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", "formal_method_baseline_comparison_decision", target_fpr)
    _write_pass_decision(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", "formal_baseline_difference_interval_decision", target_fpr)
    _write_pass_decision(run_root / "artifacts" / "external_baseline_self_containment_decision.json", "external_baseline_self_containment_decision")
    _write_pass_decision(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json", "data_split_and_leakage_guard_decision")
    _write_pass_decision(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json", "low_fpr_formal_statistics_decision")
    _write_pass_decision(run_root / "artifacts" / "paper_result_artifact_skeleton_decision.json", "paper_result_artifact_skeleton_decision", target_fpr)
    _write_pass_decision(run_root / "artifacts" / "statistical_confidence_interval_decision.json", "statistical_confidence_interval_decision")
    _write_pass_decision(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json", "validation_artifact_rebuild_dry_run_decision")

    audit = build_full_paper_result_checker_audit(run_root, config_path)

    assert audit["full_paper_result_checker_decision"] == "PASS"
    assert audit["full_paper_prompt_count"] == 2
    assert audit["full_paper_seed_per_prompt_min"] == 2
    assert audit["full_paper_calibration_unique_video_count"] == 2
    assert audit["full_paper_test_unique_video_count"] == 2
    assert audit["full_paper_runtime_attack_event_count_per_attack_min"] == 4
    assert audit["full_paper_runtime_detection_event_count_per_attack_min"] == 4
