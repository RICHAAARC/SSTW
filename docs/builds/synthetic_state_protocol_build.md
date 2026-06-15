# Synthetic State Protocol Build：Protocol Freeze + Synthetic State Inference Sanity

本文档用于指导 Codex / 工程实现阶段构建 SSTW 项目的 `B1 / SSTW Synthetic Core` 阶段。  
本阶段遵循以下总原则：

1. **工程可以前置搭建，机制必须逐阶段验证**。
2. 所有正式结果必须由 `records / thresholds / tables / reports / manifests` 重建，禁止手工拼表。
3. 所有阈值、gate 参数、融合规则只能来自 `dev / calibration`，`test` 阶段不得调参。
4. 凡是暂时无法真实实现但为了接口闭环需要预留的占位字段，字段名必须以 `_placeholder` 结尾；不可用、禁用或未运行的状态不得伪装为占位字段，应使用 `*_status`、`*_reason` 或 `*_not_run_reason` 等非占位语义字段。
5. 本项目按“假设不存在 Paper A”的绝对独立论文口径构建；`explicit_temporal_alignment` 只作为通用 baseline，不作为历史方法、前置方法或论文叙事中心。
6. 当前阶段结束时必须生成阶段 decision：`implementation_decision` 与 `mechanism_decision`，不得只用 notebook 运行成功替代机制结论。

---

## 1. 阶段定位

B1 是 SSTW 的第一个可运行阶段，负责同时完成：

```text
protocol_freeze
synthetic_state_inference_sanity
```

B1 不追求真实视频效果，也不接入真实 VAE、DiT、Flow Matching 或 trajectory。  
B1 只回答一个机制问题：

\[
\boxed{
\text{密钥条件状态空间推断是否在可控 synthetic latent 中优于普通时序聚合器与显式时间对齐 baseline？}
}
\]

B1 对应方法版本：

```text
SSTW-core-synthetic
```

B1 不得声称顶会主贡献成立，只能作为后续 B2–B6 的协议和机制 sanity foundation。

---

## 2. 阶段目标

### 2.1 必须完成

```text
1. 固定 sample role / split / threshold protocol / output layout。
2. 实现 synthetic video latent backend。
3. 实现 key-conditioned tubelet code。
4. 实现 basic state observation。
5. 实现 key-conditioned state-space synchronizer。
6. 实现 key-state admissibility gate 的最小版本。
7. 实现 synthetic temporal attacks。
8. 实现所有内部 baseline。
9. 产出 records、thresholds、tables、reports、manifest。
10. 产出 B1SyntheticStateInferenceDecision。
```

### 2.2 不得完成或不得启用

```text
real_video_vae_latent_backend
real_video_attacks
DiT_backend
Flow_Matching_backend
trajectory_observation
sampling_time_weak_constraint
external_video_watermark_baselines
```

这些模块可以存在于代码树中，但必须是：

```text
enabled = false
status = disabled
reason = "not_available_in_B1"
```

---

## 3. 推荐目录结构

```text
configs/
  protocol/
    sstw_protocol.json
    fixed_low_fpr.json
    b1_synthetic_state_inference.json
  records/
    event_record_schema.json
    state_trace_schema.json
    threshold_schema.json
  methods/
    method_variants_b1.json
  attacks/
    synthetic_temporal_attacks.json

main/
  protocol/
    calibrator.py
    record_writer.py
    manifest.py
    table_builder.py
    decision.py
  backends/
    synthetic_video_latent.py
  methods/
    state_space_watermark/
      tubelet_code.py
      state_observation.py
      key_conditioner.py
      state_synchronizer.py
      state_filter.py
      state_smoother.py
      key_state_admissibility.py
      score.py
      method_factory.py
  baselines/
    frame_prc.py
    tubelet_only.py
    explicit_temporal_alignment.py
    temporal_aggregators.py
    generic_ssm.py
  attacks/
    synthetic_temporal_attacks.py

experiments/
  b1_synthetic_state_inference/
    runner.py
    build_records.py
    mechanism_audit.py
    table_builder.py
    package_outputs.py
    README.md
```

---

## 4. Method variants

B1 必须实现以下 `method_variant`，且名称必须稳定：

```text
frame_prc
tubelet_only
explicit_temporal_alignment
generic_temporal_mean_pooling
conv1d_temporal_aggregator
gru_temporal_aggregator
transformer_temporal_aggregator
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
key_conditioned_state_space_without_admissibility
key_conditioned_state_space_without_key_condition
```

### 4.1 变体含义

| method_variant | 含义 | 是否主方法 | 目的 |
|---|---|---:|---|
| `frame_prc` | 逐帧 PRC 水印统计 | 否 | 证明逐帧迁移不足 |
| `tubelet_only` | 不做时序状态估计，只做 tubelet payload | 否 | 证明 tubelet carrier 有基本价值 |
| `explicit_temporal_alignment` | 通用 offset / scale / local search 对齐 baseline | 否 | 检验简单显式对齐是否足够 |
| `generic_temporal_mean_pooling` | 对 tubelet evidence 做均值池化 | 否 | 排除简单 pooling |
| `conv1d_temporal_aggregator` | Conv1D 时序聚合 | 否 | 排除普通局部时序模型 |
| `gru_temporal_aggregator` | GRU/LSTM 时序聚合 | 否 | 排除普通循环模型 |
| `transformer_temporal_aggregator` | Transformer 时序聚合 | 否 | 排除注意力时序聚合 |
| `generic_state_space_model` | 不含水印状态语义的 SSM | 否 | 排除套 SSM |
| `key_agnostic_state_space_model` | 有状态空间结构，但无 key condition | 否 | 证明 key condition 必要 |
| `key_conditioned_state_space_inference` | SSTW B1 主方法 | 是 | 验证密钥条件状态估计 |
| `key_conditioned_state_space_without_admissibility` | 去掉 admissibility gate | 否 | 证明 gate 防止 false positive |
| `key_conditioned_state_space_without_key_condition` | 去掉 key embedding | 否 | 证明密钥条件不是装饰 |

---

## 5. Synthetic 数据与攻击

### 5.1 Synthetic latent backend

配置字段：

```json
{
  "backend_id": "synthetic_video_latent_v1",
  "latent_distribution": "standard_gaussian",
  "latent_shape": [32, 4, 32, 32],
  "frame_count": 32,
  "latent_channels": 4,
  "latent_height": 32,
  "latent_width": 32,
  "content_id_mode": "synthetic_index",
  "prompt_id_placeholder": null,
  "generation_model_id_placeholder": "synthetic_gaussian_v1"
}
```

### 5.2 必须攻击

```text
no_attack
temporal_crop
local_clip
regular_frame_dropping
irregular_frame_dropping
frame_duplication
speed_change
frame_rate_resampling
segment_jump
latent_gaussian_noise
```

### 5.3 攻击字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `attack_name` | 攻击名称 | 否 | B2/B5 继续使用 |
| `attack_strength` | 攻击强度标识，如 crop ratio、drop rate | 否 | B2 改为真实视频攻击参数 |
| `attack_seed` | 攻击随机种子 | 否 | 保留 |
| `observed_frame_count` | 攻击后观测帧数 | 否 | 保留 |
| `temporal_mapping_gt` | synthetic 中可知的真实时间映射 | 否 | B2/B5 替换为 `temporal_mapping_gt_status` |
| `temporal_mapping_gt_status` | 真实视频中无 GT 映射状态 | 否 | B2/B5 使用 `unavailable` 等状态值 |

---

## 6. B1 records 字段设计

### 6.1 基础字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `record_version` | record schema 版本，如 `sstw_b1_v1` | 否 | 每阶段升级需显式记录 |
| `sample_id` | 样本唯一 ID | 否 | 保留 |
| `split` | `dev/calibration/test` | 否 | 保留 |
| `sample_role` | `clean_negative/attacked_negative/watermarked_positive/attacked_positive` | 否 | 保留 |
| `method_variant` | 方法变体名 | 否 | 保留 |
| `attack_name` | 攻击名 | 否 | 保留 |
| `attack_strength` | 攻击强度 | 否 | 保留 |
| `key_id` | 水印密钥 ID | 否 | 保留 |
| `content_id` | synthetic content index | 否 | B2 替换为真实视频 ID |
| `prompt_id_placeholder` | B1 无 prompt | 是 | B5 替换为真实 prompt_id |
| `seed_id` | synthetic latent seed | 否 | B5 作为 generation seed |
| `generation_model_id_placeholder` | B1 无真实生成模型 | 是 | B5 替换为真实模型 ID |
| `backend_id` | backend 名，如 `synthetic_video_latent_v1` | 否 | B2/B5 替换 |

### 6.2 Tubelet code 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `tubelet_length` | 每个 tubelet 覆盖的 latent frame 数 | 否 | 保留并消融 |
| `tubelet_spatial_patch` | 空间 patch 大小 | 否 | 保留并消融 |
| `tubelet_stride_t` | 时间 stride | 否 | B2/B5 继续使用 |
| `tubelet_stride_xy` | 空间 stride | 否 | 保留 |
| `watermark_alpha` | projection margin | 否 | B2/B5 需做质量约束消融 |
| `payload_code_id` | payload code 配置 ID | 否 | 保留 |
| `sync_code_id` | sync/reference code 配置 ID | 否 | 保留 |
| `joint_code_mode` | `payload_times_sync` 等 | 否 | 保留 |
| `embedding_mode` | B1 推荐 `projection_margin` | 否 | B5 可替换为 sampling-time constraint |

### 6.3 状态空间字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `state_model_id` | 状态模型配置 ID | 否 | 保留 |
| `state_dim` | 隐状态维度 | 否 | B3 消融 |
| `key_condition_mode` | key embedding 注入方式 | 否 | B3 重点消融 |
| `filter_mode` | `forward/filtering/bidirectional` | 否 | B3 消融 |
| `smoother_enabled` | 是否启用双向平滑 | 否 | B3 消融 |
| `phase_state_proxy` | 相位状态代理值 | 否 | B3 要求可解释 |
| `evidence_state_proxy` | 局部证据状态代理值 | 否 | B3 要求可解释 |
| `confidence_state_proxy` | 置信状态代理值 | 否 | B3 要求可解释 |
| `disturbance_state_proxy` | 时间扰动状态代理值 | 否 | B3 要求可解释 |
| `state_entropy` | 状态不确定性 | 否 | 保留 |
| `state_coverage_ratio` | 状态轨迹覆盖率 | 否 | 保留 |
| `state_matched_count` | 有效匹配 tubelet 数 | 否 | 保留 |
| `state_transition_residual` | 状态转移残差 | 否 | B3 用于 formal audit |

### 6.4 Evidence score 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `S_payload_raw` | 未状态同步的 payload 分数 | 否 | 保留 |
| `S_payload_state` | 状态同步后的 payload 分数 | 否 | 保留 |
| `S_state_posterior` | 状态空间 posterior score | 否 | 保留 |
| `S_trajectory_observation_placeholder` | B1 无 trajectory | 是 | B4 替换为 `S_trajectory_observation` |
| `S_final` | 最终检测统计量 | 否 | 保留 |
| `payload_state_gain` | `S_payload_state - S_payload_raw` | 否 | 保留 |
| `key_state_admissibility_status` | gate 状态 | 否 | 保留 |
| `negative_state_over_threshold_count` | calibration negative 中 state rescue 过阈值计数 | 否 | 保留 |

### 6.5 Threshold / decision 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `target_fpr` | 目标 FPR，如 `0.01/0.001` | 否 | 保留 |
| `threshold_id` | 阈值 ID | 否 | 保留 |
| `threshold_source_split` | 必须是 `calibration` | 否 | 保留 |
| `threshold_value` | 当前 method_variant 的阈值 | 否 | 保留 |
| `decision` | `positive/negative` | 否 | 保留 |
| `decision_reason` | 判决说明 | 否 | 保留 |
| `test_time_threshold_update_blocked` | test 阶段是否禁止调阈值 | 否 | 保留 |

---

## 7. 占位字段清理规则

B1 允许以下占位：

```text
prompt_id_placeholder
generation_model_id_placeholder
S_trajectory_observation_placeholder
trajectory_trace_placeholder
real_video_quality_metrics_placeholder
semantic_consistency_placeholder
```

禁止出现以下模糊占位：

```text
dummy
random
todo
temp
v1
placeholder_score_without_reason
```

每个占位必须有：

```text
placeholder_reason
replacement_stage
replacement_field_name
```

示例：

```json
{
  "S_trajectory_observation_placeholder": null,
  "placeholder_reason": "trajectory is disabled in B1",
  "replacement_stage": "B4",
  "replacement_field_name": "S_trajectory_observation"
}
```

---

## 8. B1 验证 gate

### 8.1 Implementation gate

```text
B1ImplementationDecision = PASS only if:
  records/event_scores.jsonl exists
  thresholds/thresholds.json exists
  all required method_variants have records
  all required attacks have records
  tables can be rebuilt from records
  no test split threshold update is observed
  all placeholder fields have replacement plan
```

### 8.2 Mechanism gate

```text
B1MechanismDecision = PASS only if:
  tubelet_only beats frame_prc on temporal attacks
  key_conditioned_state_space_inference beats generic_state_space_model on at least two complex temporal attacks
  key_conditioned_state_space_inference beats key_agnostic_state_space_model
  key_conditioned_state_space_inference beats conv1d/gru/transformer on at least one non-uniform attack
  attacked_negative_FPR <= target_fpr_tolerance
  negative_state_over_threshold_count == 0
  state_entropy is higher on failure cases than success cases or has documented explanatory trend
```

### 8.3 推荐阈值

```text
target_fpr = [0.01, 0.001]
target_fpr_tolerance = max(2 * target_fpr, target_fpr + 1 / n_attacked_negative)
```

---

## 9. B1 产物

```text
records/event_scores.jsonl
records/state_trace.jsonl
thresholds/thresholds.json
artifacts/run_manifest.json
artifacts/runtime_config.json
artifacts/b1_method_manifest.json
tables/b1_main_synthetic_table.csv
tables/b1_attack_breakdown_table.csv
tables/b1_state_model_ablation_table.csv
tables/b1_key_condition_ablation_table.csv
tables/b1_negative_safety_audit.csv
figures/b1_score_distribution.pdf
figures/b1_state_entropy_vs_failure.pdf
reports/b1_protocol_report.md
reports/b1_mechanism_audit_report.md
reports/b1_placeholder_replacement_plan.md
artifacts/B1SyntheticStateInferenceDecision.json
```

---

## 10. 进入 B2 的条件

只有当以下条件同时满足，才能进入 B2：

```text
B1ImplementationDecision = PASS
B1MechanismDecision = PASS
negative_state_over_threshold_count == 0
key_condition_ablation_gain > 0
records_rebuild_status = PASS
placeholder_replacement_plan_status = PASS
```

如果 B1 失败，应优先修复：

```text
state observation construction
key embedding injection
state entropy gate
admissibility threshold
attack strength grid
negative calibration sample size
```

不得通过调 test positive 阈值修复。
