# Real Video Latent Transfer Build：Real Video VAE Latent Transfer Check

本文档用于指导 Codex / 工程实现阶段构建 SSTW 项目的 `B2 / SSTW Real Video Latent Transfer` 阶段。  
本阶段遵循以下总原则：

1. **工程可以前置搭建，机制必须逐阶段验证**。
2. 所有正式结果必须由 `records / thresholds / tables / reports / manifests` 重建，禁止手工拼表。
3. 所有阈值、gate 参数、融合规则只能来自 `dev / calibration`，`test` 阶段不得调参。
4. 凡是暂时无法真实实现但为了接口闭环需要预留的占位字段，字段名必须以 `_placeholder` 结尾；不可用、禁用或未运行的状态不得伪装为占位字段，应使用 `*_status`、`*_reason` 或 `*_not_run_reason` 等非占位语义字段。
5. 本项目按“假设不存在 Paper A”的绝对独立论文口径构建；`explicit_temporal_alignment` 只作为通用 baseline，不作为历史方法、前置方法或论文叙事中心。
6. 当前阶段结束时必须生成阶段 decision：`implementation_decision` 与 `mechanism_decision`，不得只用 notebook 运行成功替代机制结论。

---

## 1. 阶段定位

B2 将 B1 中通过的 SSTW core 从 synthetic latent 转移到真实视频 VAE latent。  
B2 的定位是：

```text
real_video_latent_transfer_check
```

它不是顶会主实验核心，而是证明 SSTW 状态空间推断不会在真实视频 latent 中崩溃。

B2 只回答一个问题：

\[
\boxed{
\text{B1 中成立的密钥条件状态空间推断，在真实视频 VAE encode-decode-reencode 链路中是否仍然有效并保持低误报？}
}
\]

---

## 2. 阶段目标

### 2.1 必须完成

```text
1. 接入真实视频读取、裁剪、归一化、fps 处理。
2. 接入 video VAE backend。
3. 打通 source video -> VAE encode -> watermark embedding -> VAE decode -> attack -> VAE re-encode -> detection。
4. 实现真实视频时间攻击、空间攻击、压缩攻击。
5. 保留 B1 的 state-space inference 与 admissibility gate。
6. 增加质量、时序一致性、VAE reconstruction 审计。
7. 输出 B2RealVideoLatentTransferDecision。
```

### 2.2 不得启用

```text
Flow_Matching_trajectory_observation
DiT_generation_backend
sampling_time_weak_constraint
external_video_watermark_baselines_as_main_claim
```

这些可以预留接口，但必须 disabled。

---

## 3. 推荐目录结构

```text
configs/
  protocol/
    b2_real_video_latent_transfer.json
  backends/
    video_vae_backend.json
  attacks/
    real_video_attacks.json
  quality/
    quality_metrics.json
    temporal_metrics.json

main/
  backends/
    real_video_vae_latent.py
  vae/
    vae_backend.py
    vae_io.py
    vae_reconstruction_audit.py
  video/
    video_io.py
    frame_sampler.py
    fps_normalizer.py
  attacks/
    real_video_temporal_attacks.py
    compression.py
    spatial.py
    corruption.py
  analysis/
    quality_metrics.py
    temporal_metrics.py
    metric_flags.py

experiments/
  b2_real_video_latent_transfer/
    runner.py
    dataset_builder.py
    attack_runner.py
    detector_runner.py
    artifact_builder.py
    mechanism_audit.py
    table_builder.py
    package_outputs.py
```

---

## 4. 数据流程

```text
source_video
  -> frame sampling / resizing
  -> z = VAE.encode(source_video)
  -> z_wm = SSTW.embed(z, key, payload)
  -> watermarked_video = VAE.decode(z_wm)
  -> attacked_video = attack(watermarked_video)
  -> z_hat = VAE.encode(attacked_video)
  -> SSTW.detect(z_hat)
  -> fixed-FPR decision
```

negative 流程：

```text
source_video_clean
  -> attack(optional)
  -> VAE.encode
  -> detect with test key
  -> negative score distribution
```

---

## 5. B2 method variants

B2 只保留与迁移检查有关的变体：

```text
frame_prc
tubelet_only
explicit_temporal_alignment
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
key_conditioned_state_space_without_admissibility
```

不建议在 B2 放入所有 Conv1D/GRU/Transformer 变体做大规模主表。  
若 B1 中 reviewer-risk 较高，可保留：

```text
transformer_temporal_aggregator
generic_temporal_mean_pooling
```

作为轻量复核。

---

## 6. B2 攻击矩阵

必须攻击：

```text
no_attack
vae_reconstruction
h264_compression
h265_compression
spatial_resize
crop_resize
gaussian_noise
blur
temporal_crop
local_clip
regular_frame_dropping
irregular_frame_dropping
frame_duplication
speed_change
frame_rate_resampling
```

建议攻击强度采用分层：

```text
mild
medium
strong
```

每个 attack 必须写入：

```text
attack_name
attack_strength
attack_config_id
attack_seed
attack_runtime_sec
attack_failure_status
attack_failure_reason
```

---

## 7. B2 records 字段设计

### 7.1 视频与 VAE 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `source_video_id` | 原始真实视频 ID | 否 | B5 替换为 generated_video_id |
| `dataset_id` | 数据集 ID | 否 | B5 保留为 generated_dataset_id |
| `video_fps` | 处理后 fps | 否 | 保留 |
| `video_num_frames` | 处理后帧数 | 否 | 保留 |
| `video_resolution` | 分辨率 | 否 | 保留 |
| `video_duration_sec` | 视频时长 | 否 | 保留 |
| `vae_backend_id` | VAE backend ID | 否 | B5 继续使用或替换为生成模型 VAE |
| `vae_model_name` | VAE 模型名 | 否 | 保留 |
| `vae_model_version` | VAE 版本/commit/hash | 否 | 保留 |
| `vae_encode_dtype` | encode 精度 | 否 | 保留 |
| `vae_decode_dtype` | decode 精度 | 否 | 保留 |
| `vae_reconstruction_psnr` | VAE 重建 PSNR | 否，可 metric disabled | B5 保留 |
| `vae_reconstruction_ssim` | VAE 重建 SSIM | 否，可 metric disabled | B5 保留 |
| `vae_reconstruction_lpips_status` | LPIPS 指标状态 | 否 | 后续安装 LPIPS 后可从 `unavailable` 改为 `available` |
| `generation_model_id_placeholder` | B2 不是生成模型 | 是 | B5 替换为真实 `generation_model_id` |
| `prompt_id_placeholder` | B2 无 prompt | 是 | B5 替换 |

### 7.2 质量与时序一致性字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `quality_psnr` | watermarked vs source 或 attacked vs reference | 否 | 保留 |
| `quality_ssim` | 结构相似度 | 否 | 保留 |
| `quality_lpips` | 感知距离 | 可 disabled | 后续环境可用后启用 |
| `quality_metric_status` | `enabled/disabled/failed` | 否 | 保留 |
| `quality_metric_failure_reason` | metric 失败原因 | 条件字段 | 保留 |
| `temporal_flicker_score` | 帧间闪烁代理指标 | 否 | B5 可升级 |
| `motion_consistency_score` | 运动一致性代理 | 可 placeholder | B5 替换为真实 metric 或 optical flow proxy |
| `semantic_consistency_placeholder` | B2 无文本语义一致性 | 是 | B5 替换为 CLIP/VideoCLIP/text-video score |

### 7.3 Detection 字段

沿用 B1，并新增：

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `S_payload_raw` | 未同步 payload 分数 | 否 | 保留 |
| `S_payload_state` | 状态同步后 payload 分数 | 否 | 保留 |
| `S_state_posterior` | 状态 posterior score | 否 | 保留 |
| `S_trajectory_observation_placeholder` | B2 不启用 trajectory | 是 | B4 替换 |
| `S_final` | 最终分数 | 否 | 保留 |
| `state_entropy` | 状态不确定性 | 否 | 保留 |
| `state_coverage_ratio` | 覆盖率 | 否 | 保留 |
| `state_matched_count` | 匹配 tubelet 数 | 否 | 保留 |
| `state_transition_residual` | 转移残差 | 否 | B3 formal 使用 |
| `key_state_admissibility_status` | gate 状态 | 否 | 保留 |
| `negative_state_over_threshold_count` | 负样本越阈计数 | 否 | 保留 |
| `threshold_source_split` | 必须为 calibration | 否 | 保留 |

---

## 8. 占位与替换计划

B2 允许占位：

```text
prompt_id_placeholder
generation_model_id_placeholder
S_trajectory_observation_placeholder
trajectory_trace_placeholder
semantic_consistency_placeholder
motion_consistency_score_placeholder
```

替换计划：

| 占位字段 | 替换阶段 | 替换字段 |
|---|---|---|
| `prompt_id_placeholder` | B5 | `prompt_id` |
| `generation_model_id_placeholder` | B5 | `generation_model_id` |
| `S_trajectory_observation_placeholder` | B4 | `S_trajectory_observation` |
| `trajectory_trace_placeholder` | B4 | `trajectory_trace_id` / `trajectory_source` |
| `semantic_consistency_placeholder` | B5 | `semantic_consistency_score` |
| `motion_consistency_score_placeholder` | B5 | `motion_consistency_score` |

---

## 9. B2 验证 gate

### 9.1 Implementation gate

```text
B2ImplementationDecision = PASS only if:
  real videos can be encoded and decoded by VAE
  watermarked videos can be reconstructed
  all required attacks produce valid outputs or explicit failure records
  all required method variants have complete records
  thresholds are inherited/recomputed only from calibration negative
  quality and temporal metric flags are explicit
  tables can be rebuilt
```

### 9.2 Mechanism gate

```text
B2MechanismDecision = PASS only if:
  key_conditioned_state_space_inference beats tubelet_only under temporal attacks
  key_conditioned_state_space_inference beats explicit_temporal_alignment under at least one non-uniform temporal attack
  key_conditioned_state_space_inference beats key_agnostic_state_space_model
  attacked_negative_FPR <= target_fpr_tolerance
  negative_state_over_threshold_count == 0
  key_state_admissibility_status = PASS
  quality_not_collapsed = PASS
  temporal_consistency_not_collapsed = PASS
```

### 9.3 质量 gate

建议：

```text
quality_not_collapsed = PASS if:
  PSNR_drop_vs_reference <= configured_limit OR metric marked nonblocking
  SSIM_drop_vs_reference <= configured_limit OR metric marked nonblocking
  LPIPS_delta <= configured_limit OR LPIPS disabled with reason
```

### 9.4 时序 gate

```text
temporal_consistency_not_collapsed = PASS if:
  flicker_score_delta <= configured_limit
  motion_consistency_score not failed OR disabled with explicit reason
```

---

## 10. B2 产物

```text
records/event_scores.jsonl
records/state_trace.jsonl
records/quality_metrics.jsonl
thresholds/thresholds.json
artifacts/run_manifest.json
artifacts/b2_vae_manifest.json
artifacts/b2_method_manifest.json
tables/b2_real_video_latent_main_table.csv
tables/b2_attack_breakdown_table.csv
tables/b2_quality_table.csv
tables/b2_temporal_consistency_table.csv
tables/b2_state_safety_audit.csv
figures/b2_temporal_attack_curve.pdf
figures/b2_quality_robustness_tradeoff.pdf
reports/b2_real_video_latent_transfer_report.md
reports/b2_mechanism_audit_report.md
reports/b2_placeholder_replacement_plan.md
artifacts/B2RealVideoLatentTransferDecision.json
```

---

## 11. 进入 B3 的条件

```text
B2ImplementationDecision = PASS
B2MechanismDecision = PASS
quality_not_collapsed = PASS
temporal_consistency_not_collapsed = PASS
negative_state_over_threshold_count == 0
records_rebuild_status = PASS
```

如果 B2 失败，优先修复：

```text
VAE normalization / scale factor
latent shape alignment
tubelet partition on VAE latent
attack pipeline frame indexing
state coverage threshold
quality metric flags
calibration negative size
```

不得通过删除困难攻击获得 PASS；只能将攻击标记为 `unsupported_with_reason` 并在 report 中说明。
