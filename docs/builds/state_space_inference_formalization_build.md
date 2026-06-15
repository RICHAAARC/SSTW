# State Space Inference Formalization Build

本文档用于指导 Codex / 工程实现阶段构建 SSTW 项目的 `B3 / SSTW Formal Core` 阶段。  
本阶段遵循以下总原则：

1. **工程可以前置搭建，机制必须逐阶段验证**。
2. 所有正式结果必须由 `records / thresholds / tables / reports / manifests` 重建，禁止手工拼表。
3. 所有阈值、gate 参数、融合规则只能来自 `dev / calibration`，`test` 阶段不得调参。
4. 凡是暂时无法真实实现但为了接口闭环需要预留的占位字段，字段名必须以 `_placeholder` 结尾；不可用、禁用或未运行的状态不得伪装为占位字段，应使用 `*_status`、`*_reason` 或 `*_not_run_reason` 等非占位语义字段。
5. 本项目按“假设不存在 Paper A”的绝对独立论文口径构建；`explicit_temporal_alignment` 只作为通用 baseline，不作为历史方法、前置方法或论文叙事中心。
6. 当前阶段结束时必须生成阶段 decision：`implementation_decision` 与 `mechanism_decision`，不得只用 notebook 运行成功替代机制结论。

---

## 1. 阶段定位

B3 将 B1/B2 中“可运行的状态空间模块”提升为可审计、可消融、可写入论文主方法的算法原语。

B3 对应里程碑：

```text
state_space_inference_formalization
```

B3 回答的问题是：

\[
\boxed{
\text{SSTW 是否真正是密钥条件水印状态推断，而不是普通 temporal aggregator、普通 SSM 或后验分数拼接？}
}
\]

B3 结束后，应得到 SSTW 核心版本的正式冻结结论。

---

## 2. 阶段目标

### 2.1 必须完成

```text
1. 明确 observation model、transition model、key conditioner、filter、smoother、admissibility 的接口边界。
2. 使 hidden state 的 phase/evidence/confidence/disturbance 四类语义有可观测代理字段。
3. 完成时序模型对照、key condition 消融、admissibility 消融、状态变量消融。
4. 完成 unseen key、unseen attack strength、unseen attack type 泛化检查。
5. 产出 SSTW core formal decision。
```

### 2.2 不得启用

```text
trajectory_observation_as_main_score
DiT_generation_backend
sampling_time_weak_constraint
external_baselines_as_formal_proof
```

B3 仍是 SSTW core formalization，不是 SSTW-T。

---

## 3. 推荐目录结构

```text
main/
  methods/
    state_space_watermark/
      state_transition.py
      state_observation_model.py
      key_conditioner.py
      state_filter.py
      state_smoother.py
      state_variable_probes.py
      key_state_admissibility.py
      final_score.py
      formal_interface.py
  analysis/
    state_variable_ablation.py
    key_condition_ablation.py
    admissibility_audit.py
    generalization_audit.py

experiments/
  b3_state_space_formalization/
    runner.py
    ablation_builder.py
    formal_audit.py
    generalization_runner.py
    table_builder.py
    package_outputs.py

configs/
  protocol/
    b3_state_space_formalization.json
  ablations/
    state_variable_ablation.json
    key_condition_ablation.json
    temporal_model_ablation.json
    admissibility_ablation.json
```

---

## 4. 算法接口冻结

### 4.1 Observation model

输入：

```text
tubelet evidence sequence
key embedding
attack metadata for audit only
quality confidence
```

输出：

```text
x_t = [r_payload_t, r_sync_t, q_t, e_K(t)]
```

B3 中仍不启用真实 trajectory：

```text
r_traj_t_status = "disabled"
S_trajectory_observation_placeholder = null
```

### 4.2 Transition model

核心接口：

```python
h_t = transition(h_prev, x_t, key_context, transition_config)
```

必须输出：

```text
state_hidden_vector
phase_state_proxy
evidence_state_proxy
confidence_state_proxy
disturbance_state_proxy
state_transition_residual
```

### 4.3 Filter / smoother

必须支持：

```text
forward_filter
backward_filter
bidirectional_smoother
smoother_status_disabled_ablation
```

### 4.4 Admissibility gate

输入：

```text
S_payload_raw
S_payload_state
payload_state_gain
state_coverage_ratio
state_matched_count
state_entropy
calibration_negative_tail_status
```

输出：

```text
key_state_admissibility_status
admissibility_failure_reason
state_allowed_to_affect_final_score
```

---

## 5. B3 method variants

必须比较：

```text
no_state_inference
generic_temporal_mean_pooling
conv1d_temporal_aggregator
gru_temporal_aggregator
transformer_temporal_aggregator
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
key_conditioned_state_space_without_admissibility
key_conditioned_state_space_without_key_condition
key_conditioned_state_space_without_phase_state
key_conditioned_state_space_without_evidence_state
key_conditioned_state_space_without_confidence_state
key_conditioned_state_space_without_disturbance_state
key_conditioned_state_space_without_bidirectional_smoothing
key_conditioned_state_space_without_entropy_gate
```

---

## 6. B3 records 字段设计

### 6.1 Formal state 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `formal_state_schema_version` | 状态 schema 版本 | 否 | 保留 |
| `state_transition_model_id` | transition model ID | 否 | 保留 |
| `state_observation_model_id` | observation model ID | 否 | B4 增加 trajectory observation |
| `key_conditioner_id` | key conditioner 配置 | 否 | 保留 |
| `filter_mode` | forward/bidirectional 等 | 否 | 保留 |
| `smoother_mode` | smoother 类型 | 否 | 保留 |
| `phase_state_proxy` | 相位状态代理 | 否 | 保留 |
| `evidence_state_proxy` | 证据状态代理 | 否 | 保留 |
| `confidence_state_proxy` | 置信状态代理 | 否 | 保留 |
| `disturbance_state_proxy` | 扰动状态代理 | 否 | 保留 |
| `state_transition_residual` | 状态转移残差 | 否 | 保留 |
| `state_entropy` | 状态熵 | 否 | 保留 |
| `state_entropy_gate_threshold` | 熵 gate 阈值 | 否 | 保留 |
| `state_entropy_gate_status` | 熵 gate 状态 | 否 | 保留 |

### 6.2 Ablation 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `ablation_family` | `temporal_model/key_condition/state_variable/admissibility/generalization` | 否 | 保留 |
| `ablation_name` | 具体消融名 | 否 | 保留 |
| `ablation_removed_component` | 移除组件 | 否 | 保留 |
| `ablation_expected_effect` | 预期影响 | 否 | 保留 |
| `ablation_observed_delta_tpr` | TPR 变化 | 否 | 保留 |
| `ablation_observed_delta_fpr` | FPR 变化 | 否 | 保留 |
| `ablation_status` | `supports_claim/neutral/contradicts_claim` | 否 | 保留 |
| `ablation_failure_reason` | 失败说明 | 条件字段 | 保留 |

### 6.3 Generalization 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `generalization_axis` | `unseen_key/unseen_attack_strength/unseen_attack_type/unseen_content` | 否 | B5 增加 prompt/model |
| `train_condition_id` | dev/calibration 条件 | 否 | 保留 |
| `test_condition_id` | test 条件 | 否 | 保留 |
| `unseen_key_status` | unseen key 是否通过 | 否 | 保留 |
| `unseen_attack_status` | unseen attack 是否通过 | 否 | 保留 |
| `generalization_delta_tpr` | 泛化 TPR 变化 | 否 | 保留 |
| `generalization_delta_fpr` | 泛化 FPR 变化 | 否 | 保留 |

### 6.4 禁用 trajectory 字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `trajectory_enabled` | B3 必须为 false | 否 | B4 改为 true |
| `S_trajectory_observation_placeholder` | B3 无 trajectory | 是 | B4 替换 |
| `trajectory_state_adapter_placeholder` | B3 未接入 adapter | 是 | B4 替换 |

---

## 7. B3 必须证明的机制命题

### 7.1 不是普通 temporal aggregator

必须成立：

```text
key_conditioned_state_space_inference > conv1d_temporal_aggregator
key_conditioned_state_space_inference > gru_temporal_aggregator
key_conditioned_state_space_inference > transformer_temporal_aggregator
```

至少在复杂非均匀时间攻击上成立：

```text
irregular_frame_dropping
frame_duplication
frame_rate_resampling
segment_jump
very_short_local_clip
```

### 7.2 不是普通 SSM

必须成立：

```text
key_conditioned_state_space_inference > generic_state_space_model
key_conditioned_state_space_inference > key_agnostic_state_space_model
```

### 7.3 key condition 必要

必须成立：

```text
key_condition_ablation_gain > 0
unseen_key_generalization_status = PASS
wrong_key_negative_FPR <= target_fpr_tolerance
```

### 7.4 admissibility 必要

必须成立：

```text
without_admissibility_FPR >= with_admissibility_FPR
with_admissibility_negative_state_over_threshold_count == 0
admissibility_negative_tail_status = PASS
```

### 7.5 状态变量不是装饰

至少需要三类状态变量消融产生非平凡影响：

```text
phase_state
evidence_state
confidence_state
disturbance_state
bidirectional_smoothing
entropy_gate
```

---

## 8. B3 验证 gate

### 8.1 Implementation gate

```text
B3ImplementationDecision = PASS only if:
  formal state interface exists
  all ablation variants runnable
  all state proxy fields recorded
  all generalization axes have records
  all tables are rebuildable
  disabled trajectory fields are explicit
```

### 8.2 Mechanism gate

```text
B3MechanismDecision = PASS only if:
  state_space_inference_formal_decision = PASS
  key_condition_ablation_gain > 0
  admissibility_negative_tail_status = PASS
  state_variable_ablation_all_nontrivial = PASS
  unseen_key_generalization_status = PASS
  unseen_attack_generalization_status = PASS
  attacked_negative_FPR <= target_fpr_tolerance
  negative_state_over_threshold_count == 0
```

---

## 9. B3 产物

```text
records/event_scores.jsonl
records/state_trace.jsonl
records/ablation_records.jsonl
thresholds/thresholds.json
artifacts/b3_state_space_formal_manifest.json
tables/b3_state_space_main_table.csv
tables/b3_temporal_model_ablation_table.csv
tables/b3_key_condition_ablation_table.csv
tables/b3_admissibility_ablation_table.csv
tables/b3_state_variable_ablation_table.csv
tables/b3_generalization_table.csv
tables/b3_negative_tail_audit.csv
figures/b3_state_inference_diagram.pdf
figures/b3_state_entropy_distribution.pdf
figures/b3_ablation_gain_bar.pdf
reports/b3_state_space_formalization_report.md
reports/b3_mechanism_audit_report.md
reports/b3_generalization_report.md
artifacts/B3StateSpaceFormalDecision.json
```

---

## 10. 进入 B4 的条件

```text
B3ImplementationDecision = PASS
B3MechanismDecision = PASS
key_condition_ablation_gain > 0
state_variable_ablation_all_nontrivial = PASS
admissibility_negative_tail_status = PASS
negative_state_over_threshold_count == 0
trajectory_status = EXPLICIT
```

若 B3 失败，不得直接进入 trajectory；否则后续 trajectory gain 会无法解释。
