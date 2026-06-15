# Generative Video Model Probe Build

本文档用于指导 Codex / 工程实现阶段构建 SSTW 项目的 `B5 / SSTW-T Real Generative Video Validation` 阶段。  
本阶段遵循以下总原则：

1. **工程可以前置搭建，机制必须逐阶段验证**。
2. 所有正式结果必须由 `records / thresholds / tables / reports / manifests` 重建，禁止手工拼表。
3. 所有阈值、gate 参数、融合规则只能来自 `dev / calibration`，`test` 阶段不得调参。
4. 凡是暂时无法真实实现但为了接口闭环需要预留的占位字段，字段名必须以 `_placeholder` 结尾；不可用、禁用或未运行的状态不得伪装为占位字段，应使用 `*_status`、`*_reason` 或 `*_not_run_reason` 等非占位语义字段。
5. 本项目按“假设不存在 Paper A”的绝对独立论文口径构建；`explicit_temporal_alignment` 只作为通用 baseline，不作为历史方法、前置方法或论文叙事中心。
6. 当前阶段结束时必须生成阶段 decision：`implementation_decision` 与 `mechanism_decision`，不得只用 notebook 运行成功替代机制结论。

---

## 1. 阶段定位

B5 将 SSTW-T 从 trajectory core probe 推进到真实 DiT / Flow Matching 视频生成模型。  
B5 是 SSTW 顶会主论文的主实验核心。

B5 回答的问题是：

\[
\boxed{
\text{SSTW-T 是否能在真实生成式视频模型中实现低误报水印推断，并证明 trajectory observation 是生成轨迹层面的有效观测？}
}
\]

B5 不是普通真实视频 VAE latent 迁移检查。  
B5 的中心必须是：

```text
generative_video_model_probe
cross_prompt_seed_motion_generalization
quality_motion_semantic_consistency
trajectory_observation_in_real_generation
```

---

## 2. 阶段目标

### 2.1 必须完成

```text
1. 接入至少一个可运行公开视频生成模型。
2. 记录 prompt、seed、scheduler、steps、guidance、video length、resolution。
3. 打通生成视频、latent/tubelet watermark、攻击、检测、fixed-FPR 校准。
4. 获取或近似重建 generation trajectory。
5. 在真实生成视频中验证 SSTW-T。
6. 完成 prompt / seed / motion / length 泛化。
7. 完成质量、运动一致性、语义一致性审计。
8. 至少完成一个外部 baseline 或明确不可运行 limitation report。
9. 产出 B5GenerativeVideoModelDecision。
```

### 2.2 可选但强烈建议

```text
cross_generation_model_validation
external_baseline_comparison
unseen_model_generalization
```

---

## 3. 推荐目录结构

```text
main/
  generation/
    model_registry.py
    video_generator.py
    scheduler_adapter.py
    latent_capture.py
    trajectory_capture.py
    prompt_sampler.py
    seed_manager.py
    generation_manifest.py
  backends/
    generative_video_backend.py
  external_baselines/
    videomark_style_temporal_matching.py
    rivagan_adapter.py
    vidstamp_adapter.py
    baseline_registry.py
  analysis/
    semantic_consistency_audit.py
    motion_consistency_audit.py
    generation_quality_audit.py
    cross_prompt_seed_audit.py

experiments/
  b5_generative_video_model_probe/
    runner.py
    generation_runner.py
    attack_runner.py
    detection_runner.py
    external_baseline_runner.py
    generalization_runner.py
    mechanism_audit.py
    table_builder.py
    package_outputs.py

configs/
  generation/
    generation_models.json
    prompts.json
    scheduler.json
    seeds.json
  protocol/
    b5_generative_video_model_probe.json
  external_baselines/
    external_baselines.json
```

---

## 4. 模型选择规则

### 4.1 必须记录

```text
generation_model_id
generation_model_name
generation_model_family
generation_model_version
generation_model_commit_or_hash
generation_model_license_status
vae_backend_id
scheduler_id
trajectory_capture_mode
trajectory_availability_status
```

### 4.2 模型分级

```text
first_runnable_model
main_generation_model
cross_model_validation_model
```

### 4.3 选择标准

模型必须满足至少一项：

```text
can expose latent states
can expose sampling trajectory
can replay approximate trajectory
can provide video VAE latent
```

若无法记录中间状态，只能用于 external qualitative check，不能作为 B5 主模型。

---

## 5. B5 method variants

必须比较：

```text
no_watermark_or_clean_negative
frame_prc
tubelet_only
explicit_temporal_alignment
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
key_conditioned_state_space_with_trajectory
trajectory_only
trajectory_observation_without_key_condition
external_video_watermark_baselines
```

如果 B6 尚未完成，不得把 `keyed_state_trajectory_constraint` 放入 B5 主表，只能标记：

```text
keyed_state_trajectory_constraint_status_until_sampling_constraint
```

---

## 6. 外部 baseline 策略

优先级：

```text
VideoMark_style_temporal_matching
VideoMark
RivaGAN
VIDSTAMP
VideoShield
SIGMark
classical_temporal_registration
```

### 6.1 外部 baseline 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `external_baseline_name` | baseline 名称 | 否 | 保留 |
| `external_baseline_version` | 版本/commit | 条件字段 | 保留 |
| `external_baseline_runnable_status` | `runnable/not_runnable/partial` | 否 | 保留 |
| `external_baseline_not_run_reason` | 不可运行原因 | 条件字段 | 不能静默缺失 |
| `external_baseline_protocol_gap` | 协议差异说明 | 条件字段 | 保留 |
| `external_baseline_result_used_for_claim` | 是否用于正向 claim | 否 | 只有 runnable 且协议合理才为 true |

---

## 7. B5 records 字段设计

### 7.1 生成模型字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `generation_model_id` | 生成模型 ID | 否 | 保留 |
| `generation_model_name` | 模型名称 | 否 | 保留 |
| `generation_model_family` | DiT / Flow Matching / latent diffusion 等 | 否 | 保留 |
| `generation_model_version` | 版本 | 否 | 保留 |
| `generation_model_commit_or_hash` | commit/hash | 条件字段 | 保留 |
| `generation_backend_id` | backend ID | 否 | 保留 |
| `vae_backend_id` | VAE ID | 否 | 保留 |
| `scheduler_id` | scheduler ID | 否 | B6 复用 |
| `num_inference_steps` | 采样步数 | 否 | 保留 |
| `guidance_scale` | guidance 参数 | 否 | 保留 |
| `latent_capture_status` | latent 是否可记录 | 否 | 保留 |
| `trajectory_capture_status` | trajectory 是否可记录 | 否 | 保留 |
| `trajectory_capture_failure_reason` | 失败原因 | 条件字段 | 保留 |

### 7.2 Prompt / seed / motion 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `prompt_id` | prompt ID | 否 | 保留 |
| `prompt_text_hash` | prompt hash，避免日志过长 | 否 | 保留 |
| `prompt_category` | object/action/scene/motion 类型 | 否 | 保留 |
| `seed_id` | seed ID | 否 | 保留 |
| `motion_pattern_id` | 运动模式 ID | 可代理 | 保留 |
| `video_length_frames` | 生成帧数 | 否 | 保留 |
| `video_resolution` | 分辨率 | 否 | 保留 |
| `fps` | fps | 否 | 保留 |
| `heldout_prompt_status` | 是否 held-out prompt | 否 | 保留 |
| `heldout_seed_status` | 是否 held-out seed | 否 | 保留 |

### 7.3 质量、运动、语义字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `visual_quality_score` | 视觉质量指标或代理 | 否 | 保留 |
| `quality_metric_name` | 使用的质量指标 | 否 | 保留 |
| `quality_metric_status` | enabled/disabled/failed | 否 | 保留 |
| `motion_consistency_score` | 运动一致性 | 否或 disabled | 保留 |
| `motion_artifact_score` | 运动 artifact 分数 | 否或 disabled | B6 重点使用 |
| `semantic_consistency_score` | 文本/语义一致性 | 否或 disabled | 保留 |
| `semantic_metric_name` | CLIP/VideoCLIP/其他 | 否或 disabled | 保留 |
| `metric_failure_reason` | metric 失败原因 | 条件字段 | 保留 |

### 7.4 Detection / trajectory 字段

沿用 B4，并要求：

```text
S_trajectory_observation
trajectory_source
trajectory_scheduler_id
trajectory_time_grid_id
trajectory_gain_over_state_space
trajectory_negative_leakage_delta
key_state_admissibility_status
S_final
decision
```

B5 不允许：

```text
S_trajectory_observation_placeholder
generation_model_id_placeholder
prompt_id_placeholder
```

若确实缺失，必须降级为不参与 B5 主表的 invalid record。

---

## 8. B5 攻击矩阵

必须包含：

```text
no_attack
h264_compression
h265_compression
spatial_resize
crop_resize
temporal_crop
local_clip
regular_frame_dropping
irregular_frame_dropping
frame_duplication
speed_change
frame_rate_resampling
gaussian_noise
blur
```

可增强：

```text
segment_jump
mixed_temporal_spatial_attack
reencoding_pipeline_attack
```

---

## 9. B5 必须证明

### 9.1 真实生成视频低误报

```text
fixed_low_fpr_audit_pass = true
attacked_negative_FPR <= target_fpr_tolerance
negative_state_over_threshold_count == 0
```

### 9.2 状态空间优于通用 baseline

```text
key_conditioned_state_space_inference > explicit_temporal_alignment
key_conditioned_state_space_inference > generic_state_space_model
key_conditioned_state_space_inference > key_agnostic_state_space_model
```

### 9.3 trajectory 在真实生成中有独立增益

```text
key_conditioned_state_space_with_trajectory
  > key_conditioned_state_space_inference
```

同时：

```text
trajectory_control_suppression_status = PASS
trajectory_negative_leakage_delta <= tolerance
```

### 9.4 泛化

```text
cross_prompt_generalization_pass = true
cross_seed_generalization_pass = true
cross_motion_generalization_pass = true
cross_length_generalization_pass = true
```

### 9.5 质量 / 运动 / 语义

```text
quality_motion_semantic_consistency_pass = true
```

---

## 10. B5 验证 gate

### 10.1 Implementation gate

```text
B5ImplementationDecision = PASS only if:
  at least one generation model can produce videos
  generation manifest is complete
  trajectory capture or reconstruction is explicit
  all required attacks run or have failure records
  quality/motion/semantic metrics have status flags
  external baseline status is recorded
  all main records have prompt_id and generation_model_id
```

### 10.2 Mechanism gate

```text
B5MechanismDecision = PASS only if:
  generation_model_main_table_ready = true
  trajectory_observation_gain_confirmed = true
  fixed_low_fpr_audit_pass = true
  quality_motion_semantic_consistency_pass = true
  cross_prompt_seed_generalization_pass = true
  key_conditioned_state_space_with_trajectory beats explicit_temporal_alignment
```

### 10.3 顶会投稿 gate

```text
TopConferenceB5Gate = PASS only if:
  B5MechanismDecision = PASS
  B4MechanismDecision = PASS
  generative trajectory is part of main evidence
  main table centers on generation model results
  real_video_vae_latent_transfer is not the main result
```

---

## 11. B5 产物

```text
records/event_scores.jsonl
records/state_trace.jsonl
records/trajectory_trace.jsonl
records/generation_records.jsonl
records/quality_motion_semantic_records.jsonl
thresholds/thresholds.json
artifacts/b5_generation_manifest.json
artifacts/b5_model_registry_snapshot.json
tables/b5_generation_model_main_table.csv
tables/b5_temporal_attack_breakdown_table.csv
tables/b5_trajectory_gain_table.csv
tables/b5_cross_prompt_seed_generalization_table.csv
tables/b5_cross_model_generalization_table.csv
tables/b5_external_baseline_comparison_table.csv
tables/b5_quality_motion_semantic_table.csv
tables/b5_runtime_efficiency_table.csv
figures/b5_generation_attack_curve.pdf
figures/b5_trajectory_gain_by_attack.pdf
figures/b5_generated_video_case_grid.pdf
figures/b5_quality_robustness_tradeoff.pdf
reports/b5_generative_video_model_report.md
reports/b5_external_baseline_report.md
reports/b5_quality_motion_semantic_report.md
reports/b5_mechanism_audit_report.md
artifacts/B5GenerativeVideoModelDecision.json
```

---

## 12. 进入 B6 的条件

```text
B5ImplementationDecision = PASS
B5MechanismDecision = PASS
TopConferenceB5Gate = PASS
trajectory_observation_gain_confirmed = true
quality_motion_semantic_consistency_pass = true
```

如果 B5 失败，不能进入 B6 尝试用 sampling constraint 掩盖问题。应先修复：

```text
trajectory capture / reconstruction
generation latent alignment
prompt / seed instability
quality metric failures
attack pipeline
fixed-FPR negative calibration
```
