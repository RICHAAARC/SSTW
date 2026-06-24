# full_paper_result_package_gate 分阶段构建流程

本文档记录 `full_paper_result_package_gate` 的构建流程与当前完成情况。该阶段是论文结果包产出前的最后阻断 gate, 目标是保证 full_paper 在大规模 `TPR@FPR=0.001` 条件下运行时不会因为前序协议、数据、baseline、消融或攻击缺口产生阻断。

## 1. 本阶段构建流程

### 1.1 阶段目标

在真正运行 full_paper 之前, 检查所有前序阶段是否已经通过, 并冻结 full_paper 的 dataset manifest、baseline manifest、ablation manifest、attack manifest、threshold protocol 和 artifact rebuild protocol。

### 1.2 进入条件

```text
small_scale_claim_pilot_gate_passed = true
generative_video_model_probe_validation_passed = true
modern_external_baseline_records_ready = true
internal_ablation_matrix_ready = true
flow_specific_adaptive_attack_gate_passed = true
replay_and_authenticated_sketch_gate_ready_or_claim3_downgraded = true
paper_fixed_fpr_0_001_protocol_ready = true
artifact_rebuild_dry_run_passed = true
```

### 1.3 full_paper 规模要求

```text
calibration_negative_event_count >= 50000
heldout_test_negative_event_count >= 50000
heldout_attacked_positive_event_count >= 20000
negative_event_count_per_family >= 5000
attack_event_count_per_attack >= 2000
```

### 1.4 必须冻结的输入

```text
prompt_suite_manifest
seed_plan_manifest
generation_manifest
attack_manifest
baseline_manifest
ablation_manifest
threshold_manifest
artifact_rebuild_manifest
```

### 1.5 禁止事项

```text
禁止在 full_paper run 中动态新增 prompt
禁止用 held-out test 更新 threshold
禁止把 pilot records 写入主论文表格
禁止跳过 external baseline 或 adaptive attack gate
禁止把未认证 trajectory logging 当成主证据
```

### 1.6 dry-run checker 设计

full_paper 真实运行前必须先执行 dry-run checker。该 checker 不产生论文结果, 只判断 full_paper 是否允许启动。

必须输入:

```text
phase_decision_records
prompt_suite_manifest
seed_plan_manifest
generation_manifest
attack_manifest
baseline_manifest
ablation_manifest
threshold_manifest
artifact_rebuild_manifest
```

必须输出:

```text
full_paper_dry_run_decision
full_paper_allowed
blocking_stage
blocking_requirement
recommended_next_action
planned_calibration_negative_event_count
planned_heldout_test_negative_event_count
planned_attacked_positive_event_count
planned_unique_video_count
planned_event_count
```

### 1.7 统计报告要求

full_paper 结果包必须包含:

```text
binomial_confidence_interval_for_fpr
binomial_confidence_interval_for_tpr
bootstrap_confidence_interval_for_tpr_at_fpr
cluster_by_video_confidence_interval
per_negative_family_confidence_interval
per_attack_family_confidence_interval
```

若缺少这些统计报告, `submission_package_freeze` 只能输出 evidence gap, 不能输出 ready-for-submission 结论。

### 1.8 失败路径

如果 dry-run checker 失败, 只能产出 diagnostic package:

```text
reports/full_paper_blocking_report.md
artifacts/full_paper_dry_run_decision.json
manifests/full_paper_diagnostic_manifest.json
```

此时禁止生成:

```text
tables/main_detection_table.csv
tables/baseline_comparison_table.csv
reports/claim_audit_report.json
```

### 1.9 分片执行与断点续跑

full_paper 样本数量巨大, 因此本阶段必须按 shard 执行。每个 shard 必须写出:

```text
shard_id
split
prompt_id_range
seed_id_range
attack_family_range
method_variant_subset
baseline_subset
expected_record_count
actual_record_count
shard_status
shard_checksum
resume_policy
```

checker 必须确认:

```text
no_duplicate_sample_method_records
all_expected_shards_completed
all_completed_shards_schema_valid
threshold_records_written_before_heldout_evaluation
merged_records_not_written_to_checked_in_outputs
```

若 shard 缺失或重复, 只能生成 diagnostic package, 不能继续构建主表。

### 1.10 full_paper rehearsal 要求

在真正运行 full_paper 前, 必须完成 rehearsal:

```text
smoke_rehearsal_passed
pilot_rehearsal_passed
validation_rehearsal_passed
full_paper_dry_run_checker_passed
```

其中 `validation_rehearsal` 必须至少覆盖:

```text
one_external_baseline_adapter
one_internal_ablation_variant
one_adaptive_attack_family
one_replay_or_sketch_verification_path
one_artifact_rebuild_dry_run
one_confidence_interval_report
```

该要求用于避免 full_paper 在长时间运行后才发现 baseline、CI 或 packager 阻断。

### 1.11 工程实现规范索引

本阶段的 checker、统计报告和 result decision 具体接口必须遵守:

```text
docs/builds/sstw_full_paper_engineering_gate_spec.md
```

优先实现的组件为:

```text
full_paper_dry_run_checker
statistical_confidence_interval_reporter
full_paper_result_checker
```

若这些组件未实现, 本阶段只能保留为文档规范, 不能被标记为工程通过。

## 2. 当前阶段具体完成情况

### 2.1 当前完成状态

```text
stage_status: 未开始, validation-scale 前置阻塞
```

### 2.2 差距项

```text
small_scale_claim_pilot_gate 已在 workflow progression 级别 PASS
pilot_paper FPR=0.01 真实 GPU 结果尚未生成, 不能替代 full_paper 规模结果
generative_video_model_probe_validation_passed 尚未成立
modern external baseline 已进入 governed status / non-run records, 但尚无可进入主表的 runnable 同协议结果
internal ablation full-scale records 尚未完成
flow_specific_adaptive_attack_gate 尚未完成
replay_and_authenticated_sketch_gate 尚未闭合
paper-level FPR=0.001 大规模阈值协议尚未运行
full_paper_dry_run_checker 与 full_paper_result_checker 尚未实现
```

## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 未开始, validation-scale 前置阻塞 |
| 主要差距项 | small-scale pilot 已解除, 但 pilot_paper 真实结果、validation-scale、现代外部 baseline 主表对比、内部消融、adaptive attack、replay/sketch、FPR=0.001 和 full_paper checker 仍未闭合。 |
| 下一步构建方向 | 先完成 validation-scale generative probe, 同步推进现代外部 baseline adapter、内部消融、adaptive attack、replay/sketch 和 CI reporter。 |
| full_paper 影响 | 本阶段未通过时, 禁止生成 full_paper 论文结果包。 |

### 3.1 2026-06-23 最新阶段边界

最新 Wan2.1 small-scale pilot 复跑已经通过, 因此本阶段的阻塞原因不再是 pilot 未完成。当前阻塞点已经前移到 validation-scale 和论文级证据充分性:

```text
small_scale_claim_pilot_gate_passed = true
pilot_paper_result_completed = false
validation_scale_generative_probe_completed = false
modern_external_baseline_status_records_ready = true
modern_external_baseline_main_comparison_ready_count = 0
internal_ablation_full_scale_records_ready = false
flow_specific_adaptive_attack_gate_passed = false
replay_and_authenticated_sketch_gate_closed = false
paper_fixed_fpr_0_001_protocol_ready = false
full_paper_allowed = false
```

该状态允许继续实现 full_paper dry-run checker、baseline 状态审计、CI reporter 等工程组件, 但不允许生成主论文结果表或 submission package。
