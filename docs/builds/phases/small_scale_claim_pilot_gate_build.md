# small_scale_claim_pilot_gate 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段在进入大规模真实生成实验前, 以较小成本验证 Claim-1、Claim-2 和部分 Claim-3 是否有成立迹象。该阶段不产生主论文表格, 但决定是否进入 full generation 主实验。

### 1.2 输入

```text
configs/protocol/generative_video_model_probe.json
configs/protocol/sampling_time_constraint_preflight.json
configs/external_baselines/external_baselines.json
configs/generation/prompts.json
configs/generation/seeds.json
experiments/sampling_time_constraint/
experiments/generative_video_model_probe/
scripts/check_results/
scripts/package_results/
```

### 1.3 推荐规模

```text
N_prompt >= 8
N_seed_per_prompt >= 2
N_attack >= 3
N_calibration_negative_family >= 4
N_method_variant >= 6
```

### 1.4 必须覆盖的 method variant

```text
sstw_full_method
endpoint_only_control
trajectory_only_score
without_velocity_constraint
without_endpoint_aware_control
without_replay_uncertainty_weighting
generic_ssm_baseline
explicit_dtw_temporal_alignment
```

### 1.5 必须覆盖的攻击和错配

```text
video_compression
temporal_crop
frame_rate_resampling
vae_reencode_attack
wrong_sampler_replay
wrong_key_control
```

### 1.6 必须记录字段

```text
negative_family
flow_velocity_alignment_gain
path_marginal_gain_at_fixed_fpr
trajectory_payload_redundancy
quality_guard_violation_rate
negative_tail_status
wrong_key_score_separation_passed
wrong_sampler_replay_control_not_equivalent
claim_support_status
```

### 1.7 通过标准

```text
trajectory_trace_capture_success_rate >= 0.95
flow_velocity_alignment_gain > 0
path_marginal_gain_at_fixed_fpr > 0
trajectory_payload_redundancy <= preset_limit
quality_guard_violation_rate <= preset_limit
negative_tail_inflation_not_detected = true
wrong_key_score_separation_passed = true
wrong_sampler_replay_control_not_equivalent = true
```

### 1.8 失败处理

若本阶段失败, 不应进入 `generative_video_model_probe` 的 full experiment。应根据失败原因回退:

```text
flow_velocity_alignment_gain <= 0 -> 回退 sampling_time_constraint_probe
path_marginal_gain_at_fixed_fpr <= 0 -> 回退 trajectory_observation_core_probe
negative tail inflation -> 回退 protocol_governance_foundation / state_space_inference_formalization / replay_and_authenticated_sketch_gate
quality_guard_violation -> 回退 sampling_time_constraint_probe
replay failure -> 回退 replay_and_authenticated_sketch_gate 或将 Claim-3 降级
```

## 2. 当前阶段具体完成情况

### 2.1 已有工程基础

当前仓库已经存在 sampling-time constraint、generative video probe、external baseline runner、checker 与 packager 基础模块, 可组合形成小规模 pilot。

### 2.2 当前阶段缺口

当前仍需要将 pilot split、negative family、pilot-only package 和 gate 判断写入明确 runner 或 checker, 避免 pilot 结果被误用为主论文 test split 结果。

### 2.3 当前阶段使用边界

该阶段只能说明是否值得进入 full generation 主实验。它不能直接支撑主论文表格, 也不能替代 full test split 上的 fixed-FPR baseline 对比。
