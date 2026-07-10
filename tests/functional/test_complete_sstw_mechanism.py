import json
from pathlib import Path

import pytest
import torch

from experiments.generative_video_model_probe.replay_and_sketch_gate import run_replay_and_sketch_gate
from experiments.generative_video_model_probe.formal_adaptive_attack_executor import (
    run_formal_adaptive_attack_execution,
)
from main.attacks.video_runtime_attack_protocol import FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
from main.methods.state_space_watermark.authenticated_trajectory_sketch import (
    build_authenticated_trajectory_sketch_payload,
    sign_authenticated_trajectory_sketch,
    verify_authenticated_trajectory_sketch,
)
from main.methods.state_space_watermark.endpoint_latent_detector import compute_endpoint_latent_evidence
from main.methods.state_space_watermark.flow_tubelet_key_code import build_flow_tubelet_key_direction_like
from main.methods.state_space_watermark.flow_velocity_runtime import FlowVelocityConstraintRuntime
from main.methods.state_space_watermark.formal_detector import (
    apply_frozen_flow_detector,
    fit_flow_evidence_calibration,
)
from main.methods.state_space_watermark.path_observation import compute_path_step_observation
from main.methods.state_space_watermark.replay_inversion import (
    FlowSchedulePoint,
    run_flow_inversion_and_replay,
)
from main.methods.state_space_watermark.velocity_field_constraint import apply_velocity_field_constraint
from main.protocol.paper_mechanism_contract import (
    audit_paper_profile_mechanism_contract,
    load_paper_mechanism_contract,
)
from main.protocol.record_writer import write_json, write_jsonl


@pytest.mark.quick
def test_flow_tubelet_key_code_is_stable_and_key_conditioned() -> None:
    """生成、endpoint 与 replay 必须能复用同一确定性 tubelet key code。"""

    reference = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    first, first_metadata = build_flow_tubelet_key_direction_like(reference, key_text="owner-key-a")
    second, _ = build_flow_tubelet_key_direction_like(reference, key_text="owner-key-a")
    wrong, _ = build_flow_tubelet_key_direction_like(reference, key_text="owner-key-b")

    assert torch.allclose(first, second)
    assert not torch.allclose(first, wrong)
    assert torch.isclose(first.norm(), torch.tensor(1.0), atol=1e-6)
    assert first_metadata["flow_tubelet_count"] > 0


@pytest.mark.quick
def test_velocity_constraint_modifies_scheduler_model_output_with_budget() -> None:
    """弱约束必须修改 scheduler 消费的 model output, 且不突破范数预算。"""

    sample = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    model_output = torch.ones_like(sample)
    direction, _ = build_flow_tubelet_key_direction_like(sample, key_text="owner-key")

    constrained, record = apply_velocity_field_constraint(
        model_output,
        sample,
        direction,
        flow_phase=0.5,
    )

    assert not torch.allclose(constrained, model_output)
    assert record["velocity_field_source"] == "scheduler_model_output_before_flow_match_step"
    assert 0.0 < record["velocity_constraint_delta_ratio"] <= 0.02


@pytest.mark.quick
def test_path_observation_orients_velocity_to_descending_sigma_update() -> None:
    """Flow 递减 sigma 的 velocity 与实际位移必须映射到同一证据方向。"""

    sample = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    direction, _ = build_flow_tubelet_key_direction_like(sample, key_text="path-key")
    constrained_velocity = -direction
    sample_after = sample + 0.1 * direction

    observation = compute_path_step_observation(
        sample,
        sample_after,
        constrained_velocity,
        direction,
        flow_phase=0.5,
    )

    assert observation.path_projection_normalized > 0.99
    assert observation.velocity_projection_normalized > 0.99
    assert observation.path_velocity_consistency > 0.99


class FlowMatchFakeScheduler:
    """提供与 Diffusers Flow scheduler 相同的最小 step 接口。"""

    def step(self, model_output, timestep, sample, *args, **kwargs):
        del timestep, args, kwargs
        return (sample - 0.1 * model_output,)


@pytest.mark.quick
def test_flow_velocity_runtime_wraps_and_restores_scheduler_step() -> None:
    """运行时包装器必须记录真实更新并在退出 context 后恢复 scheduler。"""

    scheduler = FlowMatchFakeScheduler()
    original_step = scheduler.step
    sample = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    model_output = torch.ones_like(sample)
    with FlowVelocityConstraintRuntime(
        scheduler,
        key_text="owner-key",
        total_steps=3,
    ) as runtime:
        first = scheduler.step(model_output, torch.tensor(2.0), sample, return_dict=False)[0]
        scheduler.step(model_output, torch.tensor(1.0), first, return_dict=False)

    assert len(runtime.step_records) == 2
    assert runtime.step_records[1]["velocity_field_constraint_status"] == "applied"
    assert scheduler.step.__func__ is original_step.__func__


@pytest.mark.quick
def test_endpoint_latent_evidence_prefers_embedded_key() -> None:
    """沿正确 key direction 构造的 endpoint 必须优于错误 key。"""

    reference = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    direction, _ = build_flow_tubelet_key_direction_like(reference, key_text="embedded-key")
    endpoint = direction * 5.0 + torch.randn_like(direction) * 0.001

    matched = compute_endpoint_latent_evidence(endpoint, key_text="embedded-key")
    wrong = compute_endpoint_latent_evidence(endpoint, key_text="wrong-key")

    assert matched.score > wrong.score
    assert matched.projection > wrong.projection
    assert matched.coverage_ratio == 1.0


@pytest.mark.quick
def test_linear_flow_inversion_and_replay_closes_cycle() -> None:
    """常速度 Flow 的 reverse/forward 数值积分应恢复同一 endpoint。"""

    endpoint = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    schedule = [
        FlowSchedulePoint(timestep=2.0, sigma=1.0),
        FlowSchedulePoint(timestep=1.0, sigma=0.5),
        FlowSchedulePoint(timestep=0.0, sigma=0.0),
    ]

    def constant_velocity(latent, timestep, step_index):
        del timestep, step_index
        return torch.full_like(latent, 0.2)

    replay = run_flow_inversion_and_replay(endpoint, schedule, constant_velocity)

    assert replay.reverse_step_count == 2
    assert replay.forward_step_count == 2
    assert replay.cycle_relative_error < 1e-6


@pytest.mark.quick
def test_authenticated_trajectory_sketch_rejects_tampering() -> None:
    """HMAC sketch 必须绑定 prompt、seed、模型、sampler、时间网格和路径观测。"""

    payload = build_authenticated_trajectory_sketch_payload(
        [{
            "trajectory_step_index": 0,
            "trajectory_timestep": 1.0,
            "flow_phase": 0.5,
            "path_projection_normalized": 0.1,
            "velocity_projection_normalized": 0.2,
            "path_velocity_consistency": 0.9,
        }],
        key_id="key-id",
        prompt_digest="prompt-digest",
        seed_id="seed-a",
        model_signature="wan-model",
        sampler_signature="FlowMatchEulerDiscreteScheduler",
        time_grid_id="grid-a",
        generation_nonce_random="nonce-a",
    )
    signed = sign_authenticated_trajectory_sketch(payload, authentication_key=b"secret")

    assert verify_authenticated_trajectory_sketch(signed, authentication_key=b"secret") is True
    tampered = json.loads(json.dumps(signed))
    tampered["trajectory_sketch_payload"]["seed_id"] = "seed-b"
    assert verify_authenticated_trajectory_sketch(tampered, authentication_key=b"secret") is False


@pytest.mark.quick
def test_fixed_fpr_calibration_never_exceeds_calibration_budget() -> None:
    """冻结阈值在 `>=` 判定下也不能超过 calibration 侧目标 FPR。"""

    negatives = [
        {
            "endpoint_score": index / 20.0,
            "S_velocity": index / 40.0,
            "S_path_inv": index / 30.0,
            "path_endpoint_consistency": 0.8,
            "endpoint_coverage_ratio": 1.0,
            "replay_reliability_weight": 0.9,
            "time_grid_reliability": 0.9,
        }
        for index in range(20)
    ]
    calibration = fit_flow_evidence_calibration(
        negatives,
        method_variant="without_flow_state_admissibility",
        target_fpr=0.1,
    )
    decisions = [apply_frozen_flow_detector(record, calibration)["decision"] for record in negatives]

    assert sum(decisions) <= 2
    assert calibration.calibration_negative_count == 20


@pytest.mark.constraint
def test_all_paper_profiles_share_complete_mechanism_contract() -> None:
    """三个 paper profile 只允许统计强度与样本规模不同。"""

    contract = load_paper_mechanism_contract("configs/methods/sstw_complete_paper_mechanism.json")
    configs = [
        json.loads(Path(f"configs/protocol/{profile}_generative_probe.json").read_text(encoding="utf-8"))
        for profile in ("probe_paper", "pilot_paper", "full_paper")
    ]

    audit = audit_paper_profile_mechanism_contract(configs, contract)

    assert audit.passed is True
    assert audit.audited_profile_count == 3
    assert not audit.violations


@pytest.mark.quick
def test_full_claim3_gate_requires_real_replay_controls_and_hmac(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim-3 只有真实 replay、三类对照和 HMAC sketch 同时通过才能升级。"""

    run_root = tmp_path / "run"
    authentication_key = b"claim3-secret"
    monkeypatch.setenv("SSTW_TRAJECTORY_AUTHENTICATION_KEY", authentication_key.decode("utf-8"))
    payload = build_authenticated_trajectory_sketch_payload(
        [{
            "trajectory_step_index": 0,
            "trajectory_timestep": 1.0,
            "flow_phase": 0.5,
            "path_projection_normalized": 0.2,
            "velocity_projection_normalized": 0.3,
            "path_velocity_consistency": 0.9,
        }],
        key_id="owner-key-id",
        prompt_digest="prompt-digest",
        seed_id="seed-a",
        model_signature="wan-model",
        sampler_signature="FlowMatchEulerDiscreteScheduler",
        time_grid_id="grid-a",
        generation_nonce_random="nonce-a",
    )
    signed = sign_authenticated_trajectory_sketch(payload, authentication_key=authentication_key)
    write_jsonl(run_root / "records" / "trajectory_sketch_records.jsonl", [{
        "trajectory_trace_id": "trace-a",
        "generation_model_id": "wan-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "method_variant": "sstw_full_method",
        "authenticated_trajectory_sketch_status": "signed",
        **signed,
    }])
    write_jsonl(run_root / "records" / "formal_flow_evidence_records.jsonl", [{
        "sample_role": "attacked_positive",
        "method_variant": "sstw_full_method",
        "metric_status": "measured_formal",
        "trajectory_trace_id": "trace-a",
        "generation_model_id": "wan-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "attack_name": "codec-attack",
        "split": "test",
        "replay_inversion_status": "ready",
        "formal_flow_evidence_level": "attacked_video_wan_vae_model_velocity_replay",
        "replay_trajectory_source": "attacked_video_endpoint_model_velocity_inversion",
        "replay_uncertainty_mean": 0.1,
        "replay_reliability_weight": 0.9,
        "replay_step_counts": [16, 20, 24],
        "replay_sampler_signature": "FlowMatchEulerDiscreteScheduler",
        "authenticated_generation_time_grid_id": "grid-a",
        "replay_prompt_digest": "prompt-digest",
        "wrong_key_control_margin": 0.2,
        "wrong_prompt_control_margin": 0.1,
        "wrong_sampler_control_margin": 0.1,
        "flow_state_admissibility_status": "pass",
        "flow_posterior_confidence": 0.9,
        "flow_state_posterior_entropy": 0.1,
        "S_final_conservative": 0.8,
    }])
    write_json(run_root / "artifacts" / "three_layer_mechanism_evidence_decision.json", {
        "claim_1_velocity_constraint_detectable_watermark_decision": "PASS",
        "claim_2_path_evidence_independent_gain_decision": "PASS",
    })

    audit = run_replay_and_sketch_gate(run_root)

    assert audit["replay_and_sketch_gate_decision"] == "PASS"
    assert audit["claim3_full_support_allowed"] is True
    assert audit["replay_and_sketch_evidence_level"] == "attacked_video_wan_vae_model_velocity_replay_with_hmac_sketch"
    complete = json.loads(
        (run_root / "artifacts" / "complete_paper_mechanism_claim_decision.json").read_text(encoding="utf-8")
    )
    assert complete["complete_paper_mechanism_claim_decision"] == "PASS"


@pytest.mark.quick
def test_complete_adaptive_protocol_consumes_formal_flow_evidence(tmp_path: Path) -> None:
    """完整 profile 的11个 adaptive 协议必须来自 Flow evidence 或真实跨样本视频。"""

    run_root = tmp_path / "run"
    common = {
        "sample_role": "attacked_positive",
        "method_variant": "sstw_full_method",
        "split": "test",
        "metric_status": "measured_formal",
        "generation_model_id": "wan-model",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "trajectory_trace_id": "trace-a",
        "S_final_conservative": 0.7,
        "endpoint_score": 0.8,
        "S_path_inv": 0.5,
        "time_grid_reliability": 0.9,
        "wrong_key_control_margin": 0.2,
        "wrong_prompt_control_margin": 0.2,
        "wrong_sampler_control_margin": 0.2,
        "decision": True,
        "formal_flow_evidence_unit_id": "unit-a",
        "sstw_detector_input_contract": "formal-flow-contract",
    }
    write_jsonl(run_root / "records" / "formal_flow_evidence_records.jsonl", [
        {**common, "attack_name": "platform_transcode_runtime"},
        {**common, "attack_name": "gaussian_blur_runtime", "S_final_conservative": 0.2},
        {**common, "attack_name": "watermark_spoofing_or_copy_attack"},
        {**common, "attack_name": "collusion_multi_sample_attack"},
        {
            **common,
            "sample_role": "clean_negative",
            "attack_name": None,
            "clean_negative_video_path": "clean.mp4",
            "S_final_conservative": 0.1,
        },
    ])
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps({
        "paper_result_level": "probe_paper",
        "target_fpr": 0.1,
        "require_complete_paper_mechanism_contract": True,
        "required_non_runtime_attack_protocols": list(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS),
    }), encoding="utf-8")

    audit = run_formal_adaptive_attack_execution(run_root, config_path)
    records = [
        json.loads(line)
        for line in (run_root / "records" / "formal_adaptive_attack_execution_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert audit["formal_adaptive_attack_execution_decision"] == "PASS"
    assert len(records) == len(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS)
    assert all(record["adaptive_attack_execution_backend"] for record in records)
    assert all(record["adaptive_attack_evidence_level"] == "formal_adaptive_attack_execution" for record in records)
