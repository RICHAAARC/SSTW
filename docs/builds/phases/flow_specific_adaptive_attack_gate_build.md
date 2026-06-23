# flow_specific_adaptive_attack_gate 分阶段构建流程

本文档记录 `flow_specific_adaptive_attack_gate` 阶段的构建流程与当前完成情况。该阶段用于证明 SSTW 不是依赖固定采样步、固定 endpoint pattern 或容易被 replay mismatch 伪造的检测策略。

## 1. 本阶段构建流程

### 1.1 阶段目标

验证 SSTW 在 Flow-specific adaptive attacks 下仍能保持低误报, 并证明 time-reparameterization-invariant path evidence、replay uncertainty weighting 与 admissibility gate 对安全性有独立贡献。

### 1.2 必须覆盖的攻击

```text
scheduler_change
step_count_change
time_grid_jitter
wrong_sampler_replay
latent_noise_perturbation
vae_reencode_attack
velocity_projection_suppression
path_response_cancellation
endpoint_path_decoupling
replay_signature_mismatch
trajectory_sketch_replacement_attempt
detector_probing_with_public_negatives
```

### 1.3 必须比较

```text
sstw_full_method
without_path_integral
without_replay_uncertainty_weighting
without_admissibility
path_integral_raw
path_integral_time_normalized
path_integral_reparameterization_invariant
wrong_sampler_replay_control
wrong_time_grid_replay_control
```

### 1.4 通过标准

```text
adaptive_attack_negative_fpr_controlled = true
wrong_sampler_replay_control_not_equivalent = true
path_integral_reparameterization_invariant_beats_raw = true
without_admissibility_increases_negative_tail = true
replay_uncertainty_reduces_replay_negative_tail = true
```

### 1.5 必须记录字段

```text
adaptive_attack_name
adaptive_attack_family
adaptive_attack_strength
adaptive_attack_budget
attack_knowledge_level
targeted_evidence_layer
endpoint_preservation_status
path_response_suppression_score
velocity_projection_suppression_score
replay_signature_mismatch_status
trajectory_sketch_tamper_status
adaptive_negative_fpr
adaptive_attack_success_status
adaptive_attack_claim_support_status
```

### 1.6 审稿风险覆盖

本阶段必须能够回答以下问题:

```text
攻击者改变 scheduler 或 time grid 是否能伪造 positive?
攻击者保留 endpoint 但扰乱 path 是否能绕过 detection?
攻击者压低 velocity projection 后是否仍能保持视频质量?
攻击者替换 trajectory sketch 是否会被认证协议拒绝?
攻击者用 public negatives probe detector 是否会抬高 false positive tail?
```

未覆盖这些问题时, 论文不能声明 Flow-specific adaptive attack robustness。

### 1.7 adaptive attack 深度要求

Flow-specific adaptive attack 不能只复用普通视频攻击。至少需要覆盖三类攻击者知识:

```text
black_box_video_only_attacker
gray_box_sampler_signature_attacker
white_box_oracle_limited_flow_attacker
```

对应攻击目标至少覆盖:

```text
endpoint_preserving_path_perturbation
path_response_cancellation
time_grid_or_scheduler_mismatch
trajectory_sketch_replacement
public_negative_tail_probe
```

每类攻击都必须报告 attack budget、质量保持状态和 negative FPR。若攻击只降低 TPR 但也明显破坏视频质量, 不能被解释为成功绕过 SSTW。

### 1.8 adaptive claim 降级规则

若本阶段未通过, 论文可以报告普通视频攻击鲁棒性, 但必须禁止以下表述:

```text
robust_to_flow_specific_adaptive_attacks
robust_under_endpoint_preserving_path_attack
robust_under_scheduler_or_time_grid_adversary
```

允许的降级表述为:

```text
robustness_validated_under_non_adaptive_video_attacks
flow_specific_adaptive_robustness_requires_future_validation
```

### 1.9 adaptive attack runner 工程规范索引

本阶段 runner、records、table builder 和 checker 的接口必须遵守:

```text
docs/builds/sstw_full_paper_engineering_gate_spec.md
```

在该 runner 未实现并运行前, `flow_specific_adaptive_attack_gate` 只能保持:

```text
stage_status: 未完成
adaptive_robustness_claim_allowed: false
```

## 2. 当前阶段具体完成情况

### 2.1 当前完成状态

```text
stage_status: 未完成
```

### 2.2 已有基础

当前总体流程中已经定义该阶段。sampling-time、trajectory observation、runtime attack、runtime detection 和 small-scale pilot gate 中也已有部分 wrong-key / wrong-sampler control 字段。最新 small-scale pilot 已通过, 因此本阶段下一步可以从“等待 pilot”改为“构建 validation-scale adaptive attack manifest 与 runner”。

### 2.3 差距项

```text
缺少独立 adaptive attack runner
缺少 Flow-specific attack manifest
缺少 endpoint_path_decoupling 与 path_response_cancellation records
缺少 adaptive attack negative tail audit
缺少 full_paper 前置 checker
```

## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 未完成 |
| 主要差距项 | adaptive attack runner、manifest、negative tail audit 均未闭合。 |
| 下一步构建方向 | 在 validation-scale 中实现 adaptive attack runner、attack manifest、negative tail audit 和对应 checker。 |
| full_paper 影响 | 未通过本阶段时, full_paper 不能声明 Flow-specific adaptive attack robustness。 |

### 3.1 2026-06-23 最新阶段边界

当前 small-scale pilot 已经解除前置阻塞, 但这不等于 adaptive robustness 已成立。本阶段仍必须补齐独立的 Flow-specific adaptive attack 证据:

```text
small_scale_claim_pilot_gate_passed = true
adaptive_attack_runner_ready = false
adaptive_attack_manifest_ready = false
endpoint_path_decoupling_records_ready = false
path_response_cancellation_records_ready = false
adaptive_negative_tail_audit_ready = false
adaptive_robustness_claim_allowed = false
```

下一步应优先在 validation-scale 中构建最小可运行 adaptive attack runner, 覆盖 `scheduler_change`、`time_grid_jitter`、`wrong_sampler_replay`、`endpoint_path_decoupling` 与 `path_response_cancellation` 的受控记录。若该阶段持续缺失, full_paper 只能报告普通视频攻击鲁棒性, 不能报告 Flow-specific adaptive attack robustness。

### 3.2 2026-06-24 validation proxy runner

当前已新增 validation-scale adaptive attack proxy runner:

```text
experiments/generative_video_model_probe/adaptive_attack_runner.py
records/adaptive_attack_records.jsonl
tables/adaptive_attack_table.csv
artifacts/adaptive_attack_decision.json
reports/adaptive_attack_report.md
```

该 runner 覆盖 scheduler change、time grid jitter、wrong sampler replay、endpoint-path decoupling、path response cancellation 和 trajectory sketch replacement attempt。其作用是闭合 validation-scale 的 governed records 入口, 不是 full-paper Flow-specific adaptive robustness 证明。

当前阶段边界更新为:

```text
adaptive_attack_runner_ready = true_for_validation_proxy
adaptive_attack_manifest_ready = validation_proxy_manifest_embedded
adaptive_negative_tail_audit_ready = false
adaptive_robustness_claim_allowed = false
```

后续 full-paper 前仍需用真实 adaptive negative split、真实 quality guard 和 fixed-FPR negative tail audit 替换 validation proxy。

