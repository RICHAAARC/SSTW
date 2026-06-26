# sampling_time_constraint_probe 分阶段构建流程

本文档记录 `sampling_time_constraint_probe` 阶段的构建流程与当前完成情况。本文档只描述工程、协议、records 和 artifact 状态, 不直接支撑论文最终 claim。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段验证 sampling-time weak constraint 是否真正进入 Flow Matching 采样动力学, 并与 endpoint evidence、quality guard、semantic guard 和 flow velocity proxy 形成可审计证据链。该阶段应在 `flow_model_adapter_preflight` 确认模型 callback、time grid 与 velocity / displacement proxy 可用之后进行。

### 1.2 当前 probe 的核心比较

```text
key_conditioned_state_space_with_trajectory
keyed_state_trajectory_constraint
trajectory_constraint_without_admissibility
trajectory_constraint_without_key_condition
trajectory_constraint_wrong_key_control
```

后续 pilot 和 full experiment 还需要扩展到更多 method variant, 例如 endpoint-only、trajectory-only、without-velocity、without-replay-uncertainty 与 external baseline。

### 1.3 必须记录字段

```text
flow_velocity_proxy_available
flow_velocity_proxy_source
flow_velocity_alignment_before_constraint
flow_velocity_alignment_after_constraint
flow_velocity_alignment_gain
application_evidence_direction_cosine
constraint_application_direction_status
constraint_evidence_direction_status
latent_constraint_delta_norm
latent_norm_change
lambda_schedule_id
sampler_signature_id
sampler_signature_sha256
trajectory_source_level
```

### 1.4 通过标准

1. keyed path evidence gain 大于 unconstrained baseline。
2. keyed flow velocity alignment gain 大于 unconstrained baseline。
3. wrong-key control 不能伪造 matched-key trajectory evidence。
4. without-key control 不能伪造 matched-key trajectory evidence。
5. quality / motion / semantic guard 通过。
6. Google Drive package 可复核, 并使用 `<utc_time>_<short_commit>` 命名。

## 2. 当前阶段完成情况

### 2.1 当前阶段判定

`sampling_time_constraint_probe` 当前判定为:

```text
structure_ready / mechanism_ready / protocol_ready / artifact_ready
```

该判定基于最新 recommended 批次:

```text
package_batch_id: 20260618_023447_f325e2a5
profile: recommended
primary_model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
generation_record_count: 20
constraint_record_count: 320
```

当前 checker 输出:

```text
implementation_evidence_status: PASS
mechanism_evidence_status: PASS
missing_mechanism_requirements: []
```

### 2.2 机制层面已经完成的内容

本阶段已经完成以下机制验证:

```text
callback latent trajectory 能被捕获
sampler signature 能随 records 落盘
velocity / latent displacement proxy 能被记录
keyed sampling-time constraint 能提升 matched-key path evidence
keyed flow velocity alignment gain 大于 baseline
wrong-key control 不能伪造 matched-key trajectory evidence
without-key control 不能伪造 matched-key trajectory evidence
quality / motion / semantic guard 在 recommended 批次通过
Google Drive package 使用 <utc_time>_<short_commit> 命名并可复核
```

关键指标:

```text
keyed_constraint_alignment_gain_mean: 0.001680
keyed_flow_velocity_alignment_gain_mean: 0.020683
key_separation_gain_over_control: 0.001680
key_separation_flow_velocity_gain_over_control: 0.020685
minimum_key_separation_gain: 0.0005
minimum_key_separation_flow_velocity_gain: 0.0005
```

### 2.3 重要修复记录

本阶段曾发现 wrong-key / without-key 与 matched-key 方向不可分的问题。原因是旧方向构造使用单一正弦相位平移, 会使不同 key 的方向高度相关。当前实现已改为基于 `SHA-256(key)` 派生高维伪随机方向, 并记录:

```text
application_evidence_direction_cosine
latent_constraint_delta_norm
latent_norm_change
minimum_key_separation_gain
minimum_key_separation_flow_velocity_gain
```

recommended 批次的方向审计结果为:

```text
keyed application_evidence_direction_cosine: 1.0
without-key application_evidence_direction_cosine mean: -0.000548
wrong-key application_evidence_direction_cosine mean: -0.000084
```

### 2.4 阶段边界

本阶段可以说明 sampling-time trajectory synchronization 机制在 Wan2.1 recommended profile 上通过前置验证。它仍不能替代 small-scale claim pilot, 因为本阶段尚未覆盖完整 attack matrix、negative family、fixed-FPR path marginal gain 与 wrong-sampler replay。

下一步应进入:

```text
small_scale_mechanism_pilot_check
```



## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 已完成机制前置验证 |
| 主要差距项 | 尚未覆盖 full attack matrix、negative family、fixed-FPR path gain 和 wrong-sampler replay。 |
| 下一步构建方向 | 作为 small-scale pilot 输入, 不单独支撑 full_paper claim。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 已完成机制前置验证
gap_item: 尚未覆盖 full attack matrix、negative family、fixed-FPR path gain 和 wrong-sampler replay。
next_action: 作为 small-scale pilot 输入, 不单独支撑 full_paper claim。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```
