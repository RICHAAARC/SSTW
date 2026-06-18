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

### 2.4 motion threshold calibration 后置策略

当前 small-scale claim pilot 可以继续推进, 但 formal motion gate 使用的阈值必须按 heuristic guardrail 处理:

```text
motion_delta_threshold: 0.0005
threshold_id: motion_delta_heuristic_v1
threshold_source_split: heuristic_precalibration
usage: pilot_guardrail
```

该阈值用于 pilot 阶段阻止明显低运动视频支撑 motion-related claim, 但尚未通过大量样本统计测算。当前 pilot 的 attack、negative family、wrong-sampler replay、path marginal gain 和 replay uncertainty 可以继续推进; 但最终 paper claim 前必须补齐 `motion_threshold_calibration` 阶段。

pilot 报告中应使用以下边界表达:

```text
mechanism proxy evidence can support workflow progression.
formal motion gate currently uses heuristic_precalibration threshold.
calibrated motion threshold remains required before final paper claim.
```

### 2.5 pilot gate 自动审计器

当前已新增 small-scale claim pilot gate 自动审计入口:

```text
experiments/generative_video_model_probe/pilot_claim_gate.py
scripts/check_results/small_scale_claim_pilot_result_checker.py
```

该 checker 会从已有 governed records 中自动审计:

```text
prompt_count
seed_per_prompt_min
attack_count
negative_family_count
method_variant_count
path_marginal_gain_at_fixed_fpr
negative_tail_status
wrong_key_score_separation_passed
wrong_sampler_replay_control_not_equivalent
replay_uncertainty_mean
motion_threshold_calibration_required
```

当前 Google Drive pilot run 的 dry-run 结论为:

```text
pilot_gate_decision: FAIL
claim_support_status: workflow_progression_only
prompt_count: 8
seed_per_prompt_min: 2
quality_motion_semantic_proxy_pass: true
formal_motion_claim_status: blocked_by_formal_motion_consistency
motion_threshold_source_split: heuristic_precalibration
```

当前缺口由 checker 自动报告为:

```text
attack_matrix_ready
negative_family_ready
method_variant_ready
path_marginal_gain_ready
negative_tail_ready
wrong_key_separation_ready
wrong_sampler_replay_ready
replay_uncertainty_ready
```

这说明当前 Wan2.1 pilot 生成链路和 proxy 后处理可继续支撑 workflow progression, 但仍不能进入 full experiment 或 final paper claim。下一步应补齐 pilot postprocess / runner, 产生 attack、negative family、wrong-sampler replay 和 replay uncertainty governed records。

