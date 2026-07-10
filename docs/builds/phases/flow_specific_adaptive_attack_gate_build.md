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

当前总体流程中已经定义该阶段。sampling-time、trajectory observation、runtime attack、runtime detection 和 small-scale pilot gate 中也已有部分 wrong-key / wrong-sampler control 字段。最新 small-scale pilot 已通过, 因此本阶段下一步可以从“等待 pilot”改为“构建 probe_paper adaptive attack manifest 与 runner”。

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
| 下一步构建方向 | 在 probe_paper 中实现 adaptive attack runner、attack manifest、negative tail audit 和对应 checker。 |
| full_paper 影响 | 未通过本阶段时, full_paper 不能声明 Flow-specific adaptive attack robustness。 |

### 3.1 2026-06-23 最新阶段边界

当前正式实现已将 adaptive attack 从记录整理升级为逐视频真实优化。执行入口为：

```text
experiments/generative_video_model_probe/formal_adaptive_attack_executor.py
evaluation/attacks/adaptive_video_optimizer.py
experiments/generative_video_model_probe/adaptive_attack_runner.py
```

该实现对每个独立 held-out source video 真实解码并生成候选视频，覆盖重压缩、endpoint-preserving 扰动、探测、去除、规避以及跨视频 copy/collusion。每个候选必须记录输入输出哈希、实际查询分数、质量指标、查询预算和冻结 detector artifact。质量指标必须读取候选写盘后重新解码的帧, 避免 codec 参数尚未生效时产生虚高 PSNR。copy/spoof 以 clean recipient 视频作为 FPR 独立单位; collusion 把视频两两组成互不重叠的 pair cluster, 禁止让同一视频进入多个独立样本。不同生成模型必须使用各自 calibration split 冻结的状态空间后验与阈值。公开负样本只能用于预注册允许的探测顺序，不能更新 test 阈值或后验模型。

执行完整不等于鲁棒性主张成立。水印保留型协议必须按 source-video cluster 统计检出率, 点估计至少达到预注册最低保留率, 且95%区间下界必须高于当前目标 FPR。copy/spoof 使用 fixed-FPR 假设检验语义, 其误接受率单侧95%上界不得超过当前 profile 的目标 FPR。wrong key、prompt、sampler 与 time-grid 记录只能使用先前真实 replay control 的实测 margin, 禁止硬编码 `detected = false`。只有候选来源、保留率、spoof 拒绝和 replay control 四项同时通过, `adaptive_robustness_claim_allowed` 才能为 `true`。

正式门禁至少检查：

```text
per_video_adaptive_attack_optimization == true
adaptive_attack_query_count > 0
adaptive_attack_candidate_records_ready == true
adaptive_attack_output_video_sha256_ready == true
adaptive_attack_quality_constraint_ready == true
adaptive_attack_complete_protocol_coverage == true
adaptive_negative_tail_audit_ready == true
```

source video 是唯一独立统计单位。同一视频产生的多候选、多查询和多攻击属于簇内观测，不能扩充样本量。若真实视频文件、冻结 scorer、完整攻击覆盖或质量约束缺失，当前 paper profile 必须失败，不能用 synthetic score、validation proxy 或手工记录替代。
