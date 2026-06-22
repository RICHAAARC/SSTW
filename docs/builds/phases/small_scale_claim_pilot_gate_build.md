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
motion_threshold_calibration: PASS
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

### 2.4 motion threshold calibration 冻结策略

当前 small-scale claim pilot 可以使用已经通过的 engineering calibration threshold。pilot profile 不得用 16 条 pilot 样本重新估计 motion threshold, 否则会把独立 calibration split 覆盖为 `INSUFFICIENT_SAMPLE`。

```text
motion_threshold_calibration_decision: PASS
motion_delta_threshold: 0.010607
threshold_id: motion_delta_calibrated_v1
threshold_source_split: calibration
usage: frozen_engineering_motion_threshold_for_small_scale_pilot
```

该阈值用于 pilot 阶段阻止明显低运动视频支撑 motion-related claim。它可以解除 `blocked_until_motion_threshold_calibration`, 但仍不等价于论文级 `TPR@FPR=0.01` 或 `TPR@FPR=0.001` 证据。最终 paper claim 前仍需要更大 held-out negative split 和 frozen fixed-FPR 评估。

pilot 报告中应使用以下边界表达:

```text
mechanism proxy evidence can support small-scale pilot progression.
formal motion gate uses frozen engineering calibration threshold.
paper-level fixed-FPR evidence remains required before final claim.
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

### 2.6 pilot matrix proxy postprocess 状态

已新增 small-scale claim pilot matrix proxy 后处理入口:

```text
experiments/generative_video_model_probe/pilot_matrix_postprocess.py
```

该 runner 基于现有 generation records 与 trajectory records 构造受治理的 proxy matrix records, 覆盖:

```text
3 attacks
4 negative families
6 method variants
wrong_key separation
wrong_sampler_replay control
path_marginal_gain_at_fixed_fpr
negative_tail_status
replay_uncertainty_mean
```

当前 Google Drive pilot run 已写出:

```text
records/small_scale_claim_pilot_matrix_records.jsonl
tables/small_scale_claim_pilot_matrix_table.csv
artifacts/small_scale_claim_pilot_matrix_decision.json
reports/small_scale_claim_pilot_matrix_report.md
```

当前自动审计结果更新为:

```text
pilot_matrix_postprocess_decision: PASS
pilot_matrix_record_count: 480
pilot_matrix_attack_count: 3
pilot_matrix_method_variant_count: 6
pilot_matrix_negative_family_count: 4
path_marginal_gain_at_fixed_fpr: 0.075
replay_uncertainty_mean: 0.073608
missing_pilot_requirements: []
claim_support_status: blocked_until_motion_threshold_calibration
pilot_gate_decision: FAIL
```

该结果表示 pilot 矩阵的 proxy 后处理记录已经补齐, 但仍不能进入 full experiment 或 final paper claim。阻塞原因不再是矩阵缺失, 而是 formal motion gate 仍使用 `heuristic_precalibration` 阈值, 需要后续 `motion_threshold_calibration` 阶段。

### 2.7 runtime video-file attack runner 状态

已新增并运行真实文件级 runtime attack runner:

```text
experiments/generative_video_model_probe/attack_runner.py
```

该 runner 对已有 Wan2.1 pilot 生成视频执行实际 mp4 文件级攻击, 并写出 attacked videos 与 governed records:

```text
records/runtime_attack_records.jsonl
tables/runtime_attack_table.csv
artifacts/runtime_attack_decision.json
reports/runtime_attack_report.md
attacked_videos/
```

当前 Google Drive pilot run 的 runtime attack 结果为:

```text
runtime_attack_decision: PASS
runtime_attack_record_count: 48
runtime_attack_ready_count: 48
runtime_attack_count: 3
attack_matrix_evidence_level: runtime_video_file
claim_support_status: runtime_attack_evidence_only
```

当前 small-scale pilot gate 更新为:

```text
missing_pilot_requirements: []
attack_count: 6
runtime_attack_record_count: 48
runtime_attack_ready_count: 48
pilot_gate_decision: FAIL
claim_support_status: blocked_until_motion_threshold_calibration
```

该结果表示工程层面的 runtime attack 链路已经闭合, 但仍不能进入 final claim。剩余阻塞是 `motion_threshold_calibration`, 以及后续如需论文级攻击结论, 仍需要把 runtime attacked videos 接入正式 detection / scoring, 而不是只依赖 proxy matrix score。

### 2.8 runtime attacked video detection 闭环状态

已新增 runtime attacked video detection runner:

```text
experiments/generative_video_model_probe/detection_runner.py
```

该 runner 读取:

```text
records/runtime_attack_records.jsonl
attacked_videos/*.mp4
records/trajectory_trace.jsonl
```

并写出:

```text
records/runtime_detection_records.jsonl
tables/runtime_detection_table.csv
artifacts/runtime_detection_decision.json
reports/runtime_detection_report.md
```

当前 Google Drive pilot run 的 runtime detection 结果为:

```text
runtime_detection_decision: PASS
runtime_detection_record_count: 48
runtime_detection_ready_count: 48
runtime_detection_detectable_count: 48
runtime_detection_attack_count: 3
runtime_detection_score_mean: 0.781174
claim_support_status: runtime_detection_evidence_only
```

pilot gate 现在会显式检查: 如果存在 ready runtime attack records, 则必须存在对应 ready runtime detection records。该规则用于防止只生成 attacked videos 而未进入 detection scoring 的半闭环状态。

历史 pilot gate 在 motion calibration 未完成时仍为:

```text
pilot_gate_decision: FAIL
claim_support_status: blocked_until_motion_threshold_calibration
missing_pilot_requirements: []
```

该历史结论表示 pilot 的工程矩阵与 runtime attack / detection 链路已经闭合, 当时剩余阻塞项不是工程链路缺失, 而是 motion threshold calibration 尚未完成。最新 motion calibration 已经 PASS, 因此应重新运行 pilot profile, 让 gate 使用冻结 calibration artifact 重新计算。

### 2.9 2026-06-23 切换记录: small-scale pilot profile

Colab 入口已切换为:

```text
PROFILE: pilot
MODEL_ID: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

执行顺序为:

```text
prepare prompt suite
Wan2.1 generation with PROFILE = pilot
formal_metric_runner
reuse existing motion_threshold_calibration_decision.json
mechanism postprocess
pilot matrix postprocess
runtime attack
runtime detection
small-scale claim pilot gate
package_outputs
```

其中 `reuse existing motion_threshold_calibration_decision.json` 是关键约束。pilot profile 只读取已经通过的 calibration artifact, 不会重新运行 `motion_threshold_calibration`。
