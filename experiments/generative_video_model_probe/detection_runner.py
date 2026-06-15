"""B5 检测记录状态构建。"""

from __future__ import annotations


def build_detection_records(generation_records: list[dict], attack_records: list[dict]) -> list[dict]:
    """把生成记录与攻击矩阵合并为 detection records, 未运行时不产生正向分数。"""
    records = []
    for generation_record in generation_records:
        for attack_record in attack_records:
            records.append({
                "record_version": "generative_video_model_probe_v1",
                "generation_model_id": generation_record["generation_model_id"],
                "prompt_id": generation_record["prompt_id"],
                "seed_id": generation_record["seed_id"],
                "method_variant": "key_conditioned_state_space_with_trajectory",
                "attack_name": attack_record["attack_name"],
                "decision": "not_run",
                "decision_reason": "generation_model_not_runnable",
                "S_final": None,
                "S_trajectory_observation": None,
                "trajectory_gain_over_state_space": None,
                "trajectory_negative_leakage_delta": None,
                "negative_state_over_threshold_count": None,
            })
    return records
