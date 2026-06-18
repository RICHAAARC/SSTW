# small_scale_claim_pilot_gate 分阶段构建流程

本文档记录 `small_scale_claim_pilot_gate` 阶段的构建流程与当前完成情况。本文档只描述工程、协议、records 和 artifact 状态, 不直接支撑论文最终 claim。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段在进入大规模真实生成实验前, 以较小成本验证主要 claim 是否有成立迹象。该阶段不产生主论文最终表格, 但决定是否进入 full generation 主实验。

### 1.2 建议规模

```text
N_prompt >= 8
N_seed_per_prompt >= 2
N_attack >= 3
N_calibration_negative_family >= 4
N_method_variant >= 6
```

### 1.3 必须覆盖的攻击和错配

```text
video_compression
temporal_crop
frame_rate_resampling
vae_reencode_attack
wrong_sampler_replay
wrong_key_control
```

### 1.4 必须记录字段

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

### 1.5 通过标准

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

## 2. 当前阶段完成情况

### 2.1 当前阶段判定

`small_scale_claim_pilot_gate` 当前判定为:

```text
structure_ready / protocol_ready / external_validation_required
```

该阶段现在已经满足进入条件, 因为前置阶段已经完成:

```text
flow_model_adapter_preflight: PASS
sampling_time_constraint_probe smoke: PASS
sampling_time_constraint_probe recommended: PASS
```

最新 sampling-time recommended 证据:

```text
package_batch_id: 20260618_023447_f325e2a5
implementation_evidence_status: PASS
mechanism_evidence_status: PASS
missing_mechanism_requirements: []
```

### 2.2 pilot 必须回答的问题

pilot 不能重复证明 callback 是否能工作, 而应验证 claim 是否值得扩展到 full experiment。必须重点检查:

```text
path_marginal_gain_at_fixed_fpr > 0
negative tail 没有膨胀
wrong_sampler_replay 不能伪造正确轨迹
wrong_key / without-key control 保持分离
quality_guard 通过
attack 后 trajectory evidence 与 endpoint evidence 不发生系统性冲突
```

### 2.3 下一步建议

当前 Colab 入口已经切换到 small-scale pilot 参数:

```text
notebook: paper_workflow/colab_utils/generative_video_model_probe_colab.ipynb
PROFILE: pilot
MODEL_ID: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
CROSS_MODEL_ID: empty
prompt_limit: 8
seed_limit: 2
num_inference_steps: 16
num_frames: 49
height: 320
width: 512
run_cross_model: false
```

该设置会优先验证主模型上的 pilot 规模生成链路。attack matrix、negative family 和 wrong-sampler replay 仍需要由后续 pilot postprocess / checker 或专门 runner 继续闭合。

下一步建议按以下目标规模补齐 pilot:

```text
8 prompts
2 seeds per prompt
3 attacks
4 negative families
6 method variants
```

pilot 失败时不得进入 full experiment。pilot 通过后, 才允许进入 `generative_video_model_probe` 的完整真实模型实验。
