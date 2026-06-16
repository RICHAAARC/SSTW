# Sampling Time Constraint Submission Freeze Build：Sampling-Time Weak Constraint + Submission Freeze

本文档用于指导 Codex / 工程实现阶段构建 SSTW 项目的 `B6 / SSTW-TC + Final Paper Package` 阶段。  
本阶段遵循以下总原则：

1. **工程可以前置搭建，机制必须逐阶段验证**。
2. 所有正式结果必须由 `records / thresholds / tables / reports / manifests` 重建，禁止手工拼表。
3. 所有阈值、gate 参数、融合规则只能来自 `dev / calibration`，`test` 阶段不得调参。
4. 凡是暂时无法真实实现但为了接口闭环需要预留的占位字段，字段名必须以 `_placeholder` 结尾；不可用、禁用或未运行的状态不得伪装为占位字段，应使用 `*_status`、`*_reason` 或 `*_not_run_reason` 等非占位语义字段。
5. 本项目按“假设不存在 Paper A”的绝对独立论文口径构建；`explicit_temporal_alignment` 只作为通用 baseline，不作为历史方法、前置方法或论文叙事中心。
6. 当前阶段结束时必须生成阶段 decision：`implementation_decision` 与 `mechanism_decision`，不得只用 notebook 运行成功替代机制结论。

---

## 1. 阶段定位

B6 是 SSTW 的最高上限阶段，对应方法版本：

```text
SSTW-TC = SSTW-T + Sampling-Time Weak Constraint
```

B6 同时包含最终投稿包冻结：

```text
sampling_time_constraint_probe
submission_package_freeze
```

B6 回答两个问题：

\[
\boxed{
\text{采样过程中的弱水印约束是否能增强 trajectory-aware state observation，同时不破坏质量、运动和语义一致性？}
}
\]

以及：

\[
\boxed{
\text{SSTW / SSTW-T / SSTW-TC 的全部论文 claim 是否均可由 records 重建并通过 fixed-FPR 审计？}
}
\]

---

## 2. 阶段目标

### 2.1 Sampling-time constraint 目标

```text
1. 在生成采样过程中加入弱约束 hook。
2. 设计 lambda schedule。
3. 将 constraint 与 key-conditioned tubelet direction 对齐。
4. 验证 S_trajectory_observation 是否提升。
5. 验证 attacked positive TPR 是否提升。
6. 验证 attacked negative FPR 不上升。
7. 验证视频质量、运动一致性、语义一致性不崩。
```

### 2.2 Submission freeze 目标

```text
1. 汇总 B1–B6 全部 records。
2. 重建 thresholds、tables、figures、reports。
3. 生成 claim audit。
4. 生成 release package。
5. 明确 SSTW、SSTW-T、SSTW-TC 哪些进入主文、哪些进入附录、哪些降级为 exploratory。
```

---

## 3. 推荐目录结构

```text
main/
  generation/
    sampling_hook.py
    lambda_schedule.py
    velocity_projection_constraint.py
    constraint_controller.py
    constraint_audit.py
  analysis/
    motion_artifact_audit.py
    semantic_consistency_audit.py
    quality_delta_audit.py
    final_claim_audit.py
  packaging/
    release_builder.py
    manifest_digest.py
    rebuild_checker.py

experiments/
  b6_sampling_time_constraint/
    runner.py
    constraint_ablation_runner.py
    quality_audit_runner.py
    mechanism_audit.py
    table_builder.py
  final_submission_freeze/
    rebuild_all.py
    claim_audit.py
    package_release.py
    README.md

configs/
  generation/
    sampling_constraint.json
    lambda_schedules.json
  protocol/
    b6_sampling_time_constraint.json
    final_submission_freeze.json
```

---

## 4. B6 method variants

必须比较：

```text
key_conditioned_state_space_with_trajectory
keyed_state_trajectory_constraint
trajectory_constraint_without_admissibility
trajectory_constraint_without_key_condition
trajectory_constraint_with_lambda_schedule_ablation
trajectory_constraint_early_only
trajectory_constraint_mid_only
trajectory_constraint_late_only
trajectory_constraint_strong_lambda
trajectory_constraint_weak_lambda
```

### 4.1 变体含义

| method_variant | 含义 |
|---|---|
| `key_conditioned_state_space_with_trajectory` | SSTW-T，B6 基线 |
| `keyed_state_trajectory_constraint` | SSTW-TC 主版本 |
| `trajectory_constraint_without_admissibility` | 验证 admissibility 必要 |
| `trajectory_constraint_without_key_condition` | 验证 key condition 必要 |
| `trajectory_constraint_with_lambda_schedule_ablation` | 验证 schedule 不是随意选择 |
| `trajectory_constraint_early_only` | 早期采样约束消融 |
| `trajectory_constraint_mid_only` | 中期采样约束消融 |
| `trajectory_constraint_late_only` | 末期采样约束消融 |
| `trajectory_constraint_strong_lambda` | 强约束质量风险 |
| `trajectory_constraint_weak_lambda` | 弱约束增益边界 |

---

## 5. Sampling-time constraint 字段设计

### 5.1 Constraint 配置字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `sampling_constraint_enabled` | 是否启用采样约束 | 否 | 保留 |
| `sampling_constraint_config_id` | 约束配置 ID | 否 | 保留 |
| `constraint_projection_operator_id` | 投影算子 ID | 否 | 保留 |
| `constraint_key_id` | 约束使用的 key | 否 | 保留 |
| `constraint_payload_code_id` | payload code ID | 否 | 保留 |
| `constraint_tubelet_selector_id` | tubelet 选择策略 | 否 | 保留 |
| `lambda_schedule_id` | lambda schedule ID | 否 | 保留 |
| `lambda_max` | 最大约束强度 | 否 | 保留 |
| `lambda_time_window` | 启用时间窗口 | 否 | 保留 |
| `constraint_apply_steps` | 实际应用的采样步 | 否 | 保留 |
| `constraint_norm_budget` | 约束范数预算 | 否 | 保留 |
| `constraint_runtime_overhead_sec` | 运行开销 | 否 | 保留 |

### 5.2 Constraint 结果字段

| 字段 | 含义 | 是否占位 | 后续替换计划 |
|---|---|---:|---|
| `S_trajectory_observation_before_constraint` | 约束前 trajectory score | 否 | 保留 |
| `S_trajectory_observation_after_constraint` | 约束后 trajectory score | 否 | 保留 |
| `trajectory_constraint_gain` | 约束带来的 trajectory 增益 | 否 | 保留 |
| `attacked_positive_TPR_before_constraint` | 约束前 TPR | 否 | 保留 |
| `attacked_positive_TPR_after_constraint` | 约束后 TPR | 否 | 保留 |
| `attacked_negative_FPR_before_constraint` | 约束前 FPR | 否 | 保留 |
| `attacked_negative_FPR_after_constraint` | 约束后 FPR | 否 | 保留 |
| `quality_delta_after_constraint` | 质量变化 | 否 | 保留 |
| `motion_delta_after_constraint` | 运动一致性变化 | 否 | 保留 |
| `semantic_delta_after_constraint` | 语义一致性变化 | 否 | 保留 |
| `constraint_quality_status` | 质量 gate | 否 | 保留 |
| `constraint_motion_status` | 运动 gate | 否 | 保留 |
| `constraint_semantic_status` | 语义 gate | 否 | 保留 |
| `constraint_main_claim_status` | 是否可进入主贡献 | 否 | 保留 |

---

## 6. B6 质量 / 运动 / 语义审计

### 6.1 质量审计

必须比较：

```text
unconstrained_generation
SSTW-T_generation
SSTW-TC_generation
```

字段：

```text
visual_quality_score
quality_metric_name
quality_delta_after_constraint
quality_failure_reason
```

### 6.2 运动审计

字段：

```text
motion_consistency_score
motion_artifact_score
motion_delta_after_constraint
motion_artifact_status
motion_failure_reason
```

### 6.3 语义审计

字段：

```text
semantic_consistency_score
semantic_metric_name
semantic_delta_after_constraint
semantic_status
semantic_failure_reason
```

### 6.4 Gate

```text
quality_motion_semantic_constraint_gate = PASS only if:
  quality_delta_after_constraint <= quality_delta_limit
  motion_delta_after_constraint <= motion_delta_limit
  semantic_delta_after_constraint <= semantic_delta_limit
  motion_artifact_status != BLOCKING
```

---

## 7. B6 验证 gate

### 7.1 Implementation gate

```text
B6ImplementationDecision = PASS only if:
  sampling hook is implemented
  lambda schedule is configurable
  constraint can be toggled on/off
  all constraint ablations write records
  quality/motion/semantic audit writes records
  final package rebuild script runs
```

### 7.2 Mechanism gate

```text
B6MechanismDecision = PASS only if:
  trajectory_constraint_gain > 0
  attacked_positive_TPR_after_constraint > attacked_positive_TPR_before_constraint
  attacked_negative_FPR_after_constraint <= target_fpr_tolerance
  negative_state_over_threshold_count == 0
  quality_motion_semantic_constraint_gate = PASS
  lambda_schedule_ablation_supports_mid_stage = true
```

### 7.3 降级规则

若任一条件失败：

```text
constraint_main_claim_status = exploratory_or_appendix
```

并且主论文应回退为：

```text
SSTW-T main claim
SSTW-TC exploratory analysis
```

不得为了保留 SSTW-TC 主贡献而放宽 FPR、质量或语义 gate。

---

## 8. Final submission package 字段

### 8.1 Claim audit 字段

| 字段 | 含义 |
|---|---|
| `claim_id` | claim 唯一 ID |
| `claim_text` | 论文中的结论文本 |
| `claim_scope` | `main/appendix/exploratory` |
| `supporting_table` | 支撑表格 |
| `supporting_figure` | 支撑图 |
| `supporting_report` | 支撑报告 |
| `supporting_records_digest` | records 摘要 |
| `threshold_id` | 对应阈值 |
| `fixed_fpr_status` | 是否 fixed-FPR |
| `negative_safety_status` | negative safety 是否通过 |
| `claim_status` | `supported/unsupported/needs_downgrade` |
| `downgrade_reason` | 降级原因 |

### 8.2 Method manifest 字段

```text
method_name = "SSTW"
method_variants = ["SSTW", "SSTW-T", "SSTW-TC"]
main_variant
appendix_variants
exploratory_variants
active_evidence
disabled_evidence
generation_models
external_baselines
threshold_protocol
calibration_negative_policy
```

### 8.3 Rebuild 字段

```text
records_digest
thresholds_digest
tables_digest
figures_digest
reports_digest
package_digest
rebuild_command
rebuild_status
```

---

## 9. 最终产物

### 9.1 B6 constraint 产物

```text
records/event_scores.jsonl
records/constraint_records.jsonl
records/quality_motion_semantic_records.jsonl
thresholds/thresholds.json
artifacts/b6_sampling_constraint_manifest.json
tables/b6_sampling_constraint_main_table.csv
tables/b6_lambda_schedule_ablation_table.csv
tables/b6_constraint_quality_table.csv
tables/b6_constraint_motion_semantic_table.csv
tables/b6_constraint_negative_safety_audit.csv
figures/b6_constraint_gain_curve.pdf
figures/b6_lambda_schedule_tradeoff.pdf
figures/b6_quality_motion_semantic_tradeoff.pdf
reports/b6_sampling_time_constraint_report.md
reports/b6_quality_motion_semantic_audit_report.md
artifacts/B6SamplingTimeConstraintDecision.json
```

### 9.2 Final submission 产物

```text
records/event_scores.jsonl
records/state_trace.jsonl
records/trajectory_trace.jsonl
records/generation_records.jsonl
records/constraint_records.jsonl
thresholds/thresholds.json
artifacts/run_manifest.json
artifacts/state_space_method_manifest.json
artifacts/claim_audit.json
tables/main_fixed_fpr_table.csv
tables/temporal_attack_breakdown_table.csv
tables/state_space_ablation_table.csv
tables/key_condition_ablation_table.csv
tables/trajectory_observation_ablation_table.csv
tables/trajectory_control_table.csv
tables/generation_model_main_table.csv
tables/cross_prompt_seed_generalization_table.csv
tables/cross_model_generalization_table.csv
tables/external_baseline_comparison_table.csv
tables/quality_motion_semantic_table.csv
tables/runtime_efficiency_table.csv
tables/sampling_constraint_ablation_table.csv
figures/method_overview.pdf
figures/state_inference_diagram.pdf
figures/trajectory_observation_diagram.pdf
figures/temporal_attack_curve.pdf
figures/trajectory_score_distribution.pdf
figures/quality_robustness_tradeoff.pdf
figures/generated_video_case_grid.pdf
reports/protocol_report.md
reports/mechanism_ablation_report.md
reports/failure_case_report.md
reports/claim_audit_report.md
packages/state_space_trajectory_watermarking_release.tar.zst
```

---

## 10. 最终投稿判断

### 10.1 可投顶会最低条件

```text
B3MechanismDecision = PASS
B4MechanismDecision = PASS
B5MechanismDecision = PASS
fixed_low_fpr_audit = PASS
claim_audit = PASS
```

此时主论文版本应是：

```text
SSTW-T
```

### 10.2 强接收条件

```text
B6MechanismDecision = PASS
cross_generation_model_validation = PASS
external_baseline_comparison_ready = true
failure_case_taxonomy_ready = true
release_package_rebuildable = true
```

此时主论文可写：

```text
SSTW-TC
```

### 10.3 不建议投稿顶会的情况

```text
B4 trajectory_observation_core_probe 未通过
B5 generative_video_model_probe 未通过
fixed_low_fpr_audit 未通过
主结果仍主要来自 synthetic 或 real_video_vae_latent_transfer
external baseline 全部不可运行且无 limitation report
claim audit 中核心 claim unsupported
```

---

## 11. B6 禁止事项

```text
不得把 sampling constraint 失败结果伪装为主贡献。
不得通过降低攻击强度让 SSTW-TC 通过。
不得在 test split 上选择 lambda schedule。
不得让 trajectory score 或 state score 绕过 payload / admissibility 直接触发 positive。
不得删除 negative failure case。
不得手工修改主表。
```

---

## 12. 项目完成定义

SSTW 项目完成不是指代码能跑完，而是指：

```text
1. B1–B6 所有 enabled 阶段都有 decision。
2. 所有主 claim 均有 records 支撑。
3. 所有阈值均来自 calibration negative。
4. 所有 tables/figures/reports 均可重建。
5. SSTW、SSTW-T、SSTW-TC 的贡献边界清楚。
6. 失败模块被正确降级，而不是混入主贡献。
7. release package 可复现。
```
---

## 13. 当前工程推进状态

在 B5 recommended profile 已通过后, B6 已进入 `sampling_time_constraint_preflight` 工程阶段。当前阶段已经建立以下最小闭环:

```text
main/generation/lambda_schedule.py
main/generation/velocity_projection_constraint.py
main/generation/constraint_controller.py
experiments/sampling_time_constraint/runner.py
configs/generation/lambda_schedules.json
configs/generation/sampling_constraint.json
configs/protocol/sampling_time_constraint_preflight.json
```

当前 preflight 只验证 sampling-time weak constraint 的工程可行性、lambda schedule 消融、质量/运动/语义代理 gate 和 governed artifact 生成。它不等价于最终 B6 论文 claim, records 中必须保留:

```text
constraint_main_claim_status = preflight_only_not_final_b6_claim
submission_claim_policy = preflight_records_do_not_support_final_b6_claim
```

当前可复现命令:

```bash
python -m experiments.sampling_time_constraint.runner \
  --output-root outputs/runs/sampling_time_constraint_preflight
```

下一步应将 constraint controller 接入 Colab L4 的真实生成 sampling callback, 并复用 B5 的质量、运动和 CLIP 语义 metric 重新生成正式 records。只有真实生成链路通过后, 才能把 SSTW-TC 从 preflight / exploratory 推进到正式 B6 mechanism claim。

### 13.1 Colab L4 real sampling probe 入口

在 preflight 通过后, 已补齐 B6 Colab 真实 sampling callback probe 入口。新增文件包括:

```text
main/generation/sampling_constraint_adapter.py
experiments/sampling_time_constraint/colab_runtime.py
experiments/sampling_time_constraint/postprocess_runner.py
paper_workflow/notebook_utils/sampling_time_constraint_workflow.py
paper_workflow/colab_utils/sampling_time_constraint_colab.ipynb
scripts/package_results/sampling_time_constraint_drive_packager.py
```

Notebook 默认落盘到:

```text
/content/drive/MyDrive/SSTW/runs/sampling_time_constraint_colab
/content/drive/MyDrive/SSTW/packages/sampling_time_constraint
```

推荐首轮使用 `PROFILE = 'smoke'`, 因为该阶段首先验证 LTX callback 中修改 `latents` 是否被当前 Diffusers 版本接受。首轮通过后再切换 `recommended`。

Colab 中的执行顺序为:

```text
1. 构造 prompt suite
2. 运行 experiments.sampling_time_constraint.colab_runtime
3. 运行 formal quality / motion / CLIP semantic metric
4. 运行 experiments.sampling_time_constraint.postprocess_runner
5. 运行 pytest 与 harness
6. 打包到 packages/sampling_time_constraint
```

该 Colab probe 的正向结果只能支持:

```text
real_sampling_probe_supported_by_governed_records_not_submission_freeze
```

它仍不等同于最终 `SSTW-TC` submission freeze claim。最终 B6 claim 还需要更完整的攻击矩阵、跨 prompt / seed / model 覆盖和最终 claim audit。



---

## 14. B6 Colab 结果检查器

当前工程阶段新增了 B6 Colab 结果检查器:

```text
scripts/check_results/sampling_time_constraint_colab_result_checker.py
```

该检查器用于在 Colab L4 运行完成后显式区分三类状态:

```text
implementation_evidence_status
mechanism_evidence_status
claim_boundary
```

检查 run 目录:

```bash
python scripts/check_results/sampling_time_constraint_colab_result_checker.py   --run-root /content/drive/MyDrive/SSTW/runs/sampling_time_constraint_colab
```

检查 package 目录中的最新打包结果:

```bash
python scripts/check_results/sampling_time_constraint_colab_result_checker.py   --package-dir /content/drive/MyDrive/SSTW/packages/sampling_time_constraint
```

该检查器只允许输出 `real_sampling_probe_not_final_b6_submission_claim` 这一边界判断, 不得把 smoke / recommended probe 伪装为最终 `SSTW-TC` submission freeze claim。
