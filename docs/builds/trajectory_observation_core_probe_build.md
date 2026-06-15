# Trajectory Observation Core Probe Build

本文档用于指导 Codex / 工程实现阶段构建 SSTW 项目的 `B4 / SSTW-T Trajectory Observation` 阶段。  
本阶段遵循以下总原则：

1. **工程可以前置搭建，机制必须逐阶段验证**。
2. 所有正式结果必须由 `records / thresholds / tables / reports / manifests` 重建，禁止手工拼表。
3. 所有阈值、gate 参数、融合规则只能来自 `dev / calibration`，`test` 阶段不得调参。
4. 凡是暂时无法真实实现但为了接口闭环需要预留的占位字段，字段名必须以 `_placeholder` 结尾；不可用、禁用或未运行的状态不得伪装为占位字段，应使用 `*_status`、`*_reason` 或 `*_not_run_reason` 等非占位语义字段。
5. 本项目按“假设不存在 Paper A”的绝对独立论文口径构建；`explicit_temporal_alignment` 只作为通用 baseline，不作为历史方法、前置方法或论文叙事中心。
6. 当前阶段结束时必须生成阶段 decision：`implementation_decision` 与 `mechanism_decision`，不得只用 notebook 运行成功替代机制结论。

---

## 1. 阶段定位

B4 在已冻结的 SSTW core 上加入 generative trajectory observation。  
B4 对应方法版本：

```text
SSTW-T = SSTW + trajectory-aware state observation
```

B4 是 SSTW 进入顶会主会投稿路径的关键阶段。  
B4 回答的问题是：

\[
\boxed{
\text{生成轨迹统计是否能作为状态空间推断的独立观测项，而不是后验分数拼接或与 static tubelet evidence 冗余？}
}
\]

---

## 2. 阶段目标

### 2.1 必须完成

```text
1. 实现 trajectory trace schema。
2. 实现 detector-side trajectory observation。
3. 实现 trajectory reconstruction 或 trajectory replay 接口。
4. 实现 velocity projection statistic。
5. 将 trajectory evidence 接入 state observation x_t。
6. 实现 trajectory controls。
7. 完成 trajectory-only、without-key、generic-with-trajectory、fusion baseline。
8. 证明 trajectory 对 state-space inference 有独立边际增益。
9. 产出 B4TrajectoryObservationDecision。
```

### 2.2 不得完成或不得启用

```text
sampling_time_weak_constraint
trajectory_constraint_embedding
final_generation_claim_without_B5
```

B4 可以使用 replay / surrogate / approximate inversion，但必须在字段中说明 trajectory source。  
如果 trajectory source 不是真实生成采样轨迹，则不能写成主实验结论，只能作为 core probe。

---

## 3. 推荐目录结构

```text
main/
  trajectory/
    trajectory_trace.py
    trajectory_reconstruction.py
    trajectory_observation.py
    trajectory_statistic.py
    velocity_projection.py
    trajectory_controls.py
    trajectory_runtime.py
  methods/
    state_space_watermark/
      trajectory_state_observation.py
      trajectory_state_adapter.py
      final_score_with_trajectory.py

experiments/
  b4_trajectory_observation_core/
    runner.py
    trajectory_builder.py
    control_runner.py
    mechanism_audit.py
    correlation_audit.py
    table_builder.py
    package_outputs.py

configs/
  trajectory/
    trajectory_observation.json
    trajectory_controls.json
    trajectory_time_grid.json
  protocol/
    b4_trajectory_observation_core.json
```

---

## 4. Trajectory source 分级

B4 必须支持以下 trajectory source 类型：

```text
recorded_sampling_trace
approximate_inversion_trace
latent_replay_trace
synthetic_surrogate_trace
```

### 4.1 允许作为 B4 正式机制证据的 source

```text
recorded_sampling_trace
approximate_inversion_trace
latent_replay_trace
```

### 4.2 只能作为 smoke / debug 的 source

```text
synthetic_surrogate_trace
```

### 4.3 禁止静默缺失

如果无法获得 trajectory，必须写：

```text
trajectory_source = null
trajectory_source_status = "unavailable"
trajectory_status_reason = "..."
S_trajectory_observation = null
trajectory_enabled = false
```

不得用 0 分数伪装为真实 trajectory。

---

## 5. B4 method variants

必须比较：

```text
key_conditioned_state_space_inference
key_conditioned_state_space_with_trajectory
trajectory_only
trajectory_observation_without_key_condition
generic_state_space_with_trajectory
explicit_temporal_alignment_with_trajectory_fusion
trajectory_late_score_fusion
trajectory_random_key_control
trajectory_time_shuffled_control
trajectory_direction_shuffled_control
```

### 5.1 变体目的

| method_variant | 目的 |
|---|---|
| `key_conditioned_state_space_inference` | SSTW core，不含 trajectory |
| `key_conditioned_state_space_with_trajectory` | SSTW-T 主方法 |
| `trajectory_only` | 检查 trajectory 单独是否足够，通常不应作为最终 positive 唯一来源 |
| `trajectory_observation_without_key_condition` | 证明 key condition 必要 |
| `generic_state_space_with_trajectory` | 排除普通 SSM + trajectory |
| `explicit_temporal_alignment_with_trajectory_fusion` | 排除显式对齐 + 后验 trajectory 加权 |
| `trajectory_late_score_fusion` | 证明不是分数拼接 |
| `trajectory_random_key_control` | 防止随机 key 泄漏 |
| `trajectory_time_shuffled_control` | 证明时间结构必要 |
| `trajectory_direction_shuffled_control` | 证明投影方向结构必要 |

---

## 6. B4 records 字段设计

### 6.1 Trajectory trace 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `trajectory_enabled` | 是否启用 trajectory | 否 | B5 保留 |
| `trajectory_source` | trace 来源 | 否 | B5 目标为真实生成 trace |
| `trajectory_source_status` | `valid/approximate/surrogate/unavailable` | 否 | B5 必须尽量 valid |
| `trajectory_status_reason` | 不可用原因 | 条件字段 | B5 降低此类样本比例 |
| `trajectory_trace_id` | trace 唯一 ID | 否，若 enabled | 保留 |
| `trajectory_time_grid_id` | 时间网格配置 | 否 | B5 与 scheduler 对齐 |
| `trajectory_num_steps` | 轨迹采样步数 M | 否 | B5 保留 |
| `trajectory_time_points` | 采样时间点列表或摘要 | 否 | 保留 |
| `trajectory_scheduler_id_placeholder` | B4 若非真实 scheduler，则占位 | 可占位 | B5 替换为真实 scheduler_id |
| `velocity_estimator_id` | 速度估计方法 | 否 | 保留 |
| `velocity_projection_operator_id` | 投影算子 | 否 | B6 复用 |
| `trajectory_runtime_sec` | 轨迹处理时间 | 否 | 保留 |

### 6.2 Trajectory score 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `S_trajectory_observation` | trajectory 观测分数 | 否，若 enabled | 保留 |
| `S_traj_state` | trajectory 接入状态后的分数 | 否 | 保留 |
| `trajectory_state_gain` | `SSTW-T - SSTW` 的分数增益 | 否 | 保留 |
| `trajectory_gain_over_state_space` | TPR 或 score 层面的增益 | 否 | B5 主表使用 |
| `trajectory_negative_leakage_delta` | 加入 trajectory 后 negative tail 变化 | 否 | 保留 |
| `trajectory_payload_correlation` | 与 `S_payload_state` 相关性 | 否 | 保留 |
| `trajectory_state_correlation` | 与 `S_state_posterior` 相关性 | 否 | 保留 |
| `trajectory_control_suppression_status` | control 是否被抑制 | 否 | 保留 |
| `trajectory_control_failure_reason` | control 失败原因 | 条件字段 | 保留 |

### 6.3 Trajectory control 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `control_type` | `random_key/time_shuffle/direction_shuffle/noise_trace` | 否 | 保留 |
| `control_expected_effect` | control 预期效果 | 否 | 保留 |
| `control_observed_score` | control 分数 | 否 | 保留 |
| `control_delta_vs_main` | 与主 trajectory 的差异 | 否 | 保留 |
| `control_status` | `suppressed/not_suppressed` | 否 | 保留 |
| `control_not_run_reason` | 未运行原因 | 条件字段 | B5 应减少未运行 |

---

## 7. 占位与替换计划

B4 允许：

```text
trajectory_scheduler_id_placeholder
generation_model_id_placeholder
prompt_id_placeholder
semantic_consistency_placeholder
sampling_constraint_placeholder
```

替换计划：

| 占位字段 | 替换阶段 | 替换字段 |
|---|---|---|
| `trajectory_scheduler_id_placeholder` | B5 | `trajectory_scheduler_id` |
| `generation_model_id_placeholder` | B5 | `generation_model_id` |
| `prompt_id_placeholder` | B5 | `prompt_id` |
| `semantic_consistency_placeholder` | B5 | `semantic_consistency_score` |
| `sampling_constraint_placeholder` | B6 | `sampling_constraint_config_id` |

---

## 8. B4 必须证明

### 8.1 统计分离

```text
trajectory_positive_distribution > trajectory_negative_distribution
```

必须体现在：

```text
tables/b4_trajectory_score_distribution_table.csv
figures/b4_trajectory_score_distribution.pdf
```

### 8.2 非冗余

必须满足：

```text
abs(correlation(S_trajectory_observation, S_payload_state)) < correlation_threshold
abs(correlation(S_trajectory_observation, S_state_posterior)) < correlation_threshold
```

若相关性高，则需要证明 trajectory 在特定攻击下仍有 conditional gain。

### 8.3 fixed-FPR 增益

必须成立：

```text
key_conditioned_state_space_with_trajectory
  > key_conditioned_state_space_inference
```

且：

```text
trajectory_negative_leakage_delta <= 0 or within tolerance
attacked_negative_FPR <= target_fpr_tolerance
```

### 8.4 control 抑制

必须成立：

```text
trajectory_random_key_control < main trajectory score
trajectory_time_shuffled_control < main trajectory score
trajectory_direction_shuffled_control < main trajectory score
```

---

## 9. B4 验证 gate

### 9.1 Implementation gate

```text
B4ImplementationDecision = PASS only if:
  trajectory trace schema exists
  trajectory source is explicit for every sample
  trajectory observation is computed for enabled samples
  all controls are runnable or have not_run_reason
  trajectory-state adapter writes records
  score correlation table can be rebuilt
  runtime overhead is measured
```

### 9.2 Mechanism gate

```text
B4MechanismDecision = PASS only if:
  trajectory_observation_mechanism_decision = PASS
  trajectory_gain_over_state_space > 0
  trajectory_negative_leakage_delta <= tolerance
  correlation_status = PASS
  control_suppression_status = PASS
  runtime_overhead_status != BLOCKING
```

### 9.3 顶会最低 gate

```text
TopConferenceTrajectoryGate = PASS only if:
  B4MechanismDecision = PASS
  trajectory source is not only synthetic_surrogate_trace
  trajectory gain appears in fixed-FPR attacked positive setting
  trajectory controls are suppressed
```

---

## 10. B4 产物

```text
records/event_scores.jsonl
records/state_trace.jsonl
records/trajectory_trace.jsonl
records/trajectory_control_records.jsonl
thresholds/thresholds.json
artifacts/b4_trajectory_manifest.json
tables/b4_trajectory_main_table.csv
tables/b4_trajectory_ablation_table.csv
tables/b4_trajectory_control_table.csv
tables/b4_score_correlation_table.csv
tables/b4_runtime_table.csv
figures/b4_trajectory_score_distribution.pdf
figures/b4_trajectory_gain_by_attack.pdf
figures/b4_score_correlation_heatmap.pdf
reports/b4_trajectory_observation_report.md
reports/b4_trajectory_control_report.md
reports/b4_mechanism_audit_report.md
artifacts/B4TrajectoryObservationDecision.json
```

---

## 11. 进入 B5 的条件

```text
B4ImplementationDecision = PASS
B4MechanismDecision = PASS
TopConferenceTrajectoryGate = PASS
trajectory_gain_over_state_space > 0
control_suppression_status = PASS
trajectory_source_status != only_surrogate
```

若 B4 失败，不得直接接入真实生成模型来“赌结果”。应先修复：

```text
trajectory time grid
velocity estimator
trajectory projection direction
trajectory-state adapter
trajectory controls
correlation with static evidence
runtime bottleneck
```
