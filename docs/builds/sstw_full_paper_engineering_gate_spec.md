# SSTW full_paper 工程门禁实现规范

## 0. 文档定位

本文档用于把 `sstw_project_construction_flow.md`、`sstw_top_tier_experimental_sufficiency_checklist.md` 和各阶段构建文档中的 full_paper 门禁要求, 转换为 Codex 可实现的 repository checker、runner、reporter 和 builder 接口规范。

本文档不记录当前项目完成状态, 不生成实验结果, 不替代任何 governed records。当前阶段状态仍应记录在:

```text
docs/builds/sstw_phase_completion_status.md
docs/builds/phases/
```

本文档的核心目标是:

```text
让 Codex 在后续工程实现时, 能按固定接口逐步实现 full_paper dry-run checker、full_paper result checker、modern baseline runner、adaptive attack runner、statistical CI reporter 和 reviewer evidence index builder。
```

## 1. 总体工程门禁架构

full_paper 工程门禁分为六类组件:

| 组件 | 作用 | 产物性质 |
|---|---|---|
| `full_paper_dry_run_checker` | 判断 full_paper 是否允许启动 | gate decision |
| `modern_external_baseline_runner` | 运行或登记现代外部 baseline | governed records |
| `flow_specific_adaptive_attack_runner` | 运行 Flow-specific adaptive attacks | governed records |
| `statistical_confidence_interval_reporter` | 计算低 FPR、TPR 和 cluster-by-video 置信区间 | governed reports |
| `full_paper_result_checker` | 检查 full_paper records、tables、figures、reports 是否满足论文主张 | gate decision |
| `reviewer_evidence_index_builder` | 将审稿问题映射到 governed artifacts | governed report |

这些组件属于通用科研工程门禁。SSTW 的项目特定要求在于: 所有组件必须围绕 Flow Matching 轨迹水印的 state-space evidence、path evidence、endpoint evidence、replay uncertainty 和 fixed low-FPR calibration 展开。

## 2. 推荐实现位置

推荐实现路径如下:

```text
scripts/check_results/full_paper_dry_run_checker.py
scripts/check_results/full_paper_result_checker.py
experiments/generative_video_model_probe/external_baseline_runner.py
experiments/flow_specific_adaptive_attack/runner.py
experiments/flow_specific_adaptive_attack/table_builder.py
experiments/flow_specific_adaptive_attack/package_outputs.py
main/analysis/statistical_confidence_intervals.py
experiments/submission_freeze_preparation/reviewer_evidence_index.py
tests/functional/test_full_paper_dry_run_checker.py
tests/functional/test_full_paper_result_checker.py
tests/functional/test_statistical_confidence_intervals.py
tests/functional/test_reviewer_evidence_index.py
```

若后续实现中选择其他路径, 必须保持语义名称清晰, 并同步更新阶段文档和 harness 可见的路径规则。

## 3. full_paper_dry_run_checker 接口规范

### 3.1 输入

`full_paper_dry_run_checker` 只能读取 planning artifacts, 不读取 held-out test 分数。

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
resource_budget_manifest
shard_plan_manifest
```

可选输入:

```text
previous_validation_summary
external_baseline_preflight_summary
adaptive_attack_preflight_summary
replay_or_sketch_gate_summary
```

### 3.2 禁止输入

```text
heldout_test_event_scores
heldout_test_detection_scores
manual_table_values
manually_selected_best_threshold
```

这些输入若出现, checker 必须失败, 因为 dry-run 阶段不得接触 test split 检测结果。

### 3.3 输出

必须输出:

```text
artifacts/full_paper_dry_run_decision.json
reports/full_paper_blocking_report.md
manifests/full_paper_dry_run_manifest.json
```

`full_paper_dry_run_decision.json` 必须至少包含:

```text
full_paper_dry_run_decision
full_paper_allowed
blocking_stage
blocking_requirement
observed_value
required_value
recommended_next_action
claim_support_status
planned_calibration_negative_event_count
planned_heldout_test_negative_event_count
planned_attacked_positive_event_count
planned_unique_video_count
planned_event_count
shard_plan_status
resource_budget_status
```

### 3.4 PASS 条件

只有同时满足以下条件, 才能输出 `full_paper_allowed = true`:

```text
all_required_phase_decisions_exist
all_required_phase_decisions_passed_or_downgraded
small_scale_claim_pilot_gate_passed
validation_scale_probe_passed
modern_external_baseline_plan_complete
internal_ablation_plan_complete
adaptive_attack_plan_complete
replay_or_claim3_downgrade_plan_complete
threshold_manifest_frozen
prompt_suite_manifest_frozen
calibration_negative_event_count_plan_sufficient
heldout_test_negative_event_count_plan_sufficient
artifact_rebuild_plan_complete
resource_budget_status == ready
checked_in_outputs_blocked
```

## 4. modern_external_baseline_runner 接口规范

### 4.1 baseline 分层

runner 必须支持至少三种 baseline 状态:

```text
runnable
non_runnable_with_governed_reason
protocol_incompatible_with_governed_reason
```

不得静默删除现代 baseline。无法运行时也必须写出 governed record。

### 4.2 记录字段

每个 baseline 必须写出:

```text
external_baseline_name
external_baseline_source_url
external_baseline_family
external_baseline_runnable_status
external_baseline_adapter_status
external_baseline_input_compatibility_status
external_baseline_output_record_status
external_baseline_threshold_policy_compatible
external_baseline_attack_manifest_compatible
external_baseline_protocol_gap
external_baseline_not_run_reason
external_baseline_result_used_for_claim
```

其中, `external_baseline_result_used_for_claim` 只有在 baseline 真实运行且协议兼容时才允许为 true。

### 4.3 主表准入

进入主表的 baseline 必须满足:

```text
external_baseline_runnable_status == runnable
external_baseline_adapter_status == ready
external_baseline_output_record_status == governed_records_written
external_baseline_threshold_policy_compatible == true
external_baseline_attack_manifest_compatible == true
external_baseline_result_used_for_claim == true
```

若只引用外部论文数字, 只能进入 related work 或 supplementary discussion, 不能进入主表胜负判断。

## 5. flow_specific_adaptive_attack_runner 接口规范

### 5.1 攻击者知识层级

runner 至少应覆盖:

```text
black_box_video_only_attacker
gray_box_sampler_signature_attacker
white_box_oracle_limited_flow_attacker
```

### 5.2 攻击目标

至少应覆盖:

```text
endpoint_preserving_path_perturbation
path_response_cancellation
velocity_projection_suppression
time_grid_or_scheduler_mismatch
trajectory_sketch_replacement
public_negative_tail_probe
```

### 5.3 必须记录字段

```text
adaptive_attack_name
adaptive_attack_family
adaptive_attack_strength
adaptive_attack_budget
attack_knowledge_level
targeted_evidence_layer
endpoint_preservation_status
path_response_suppression_score
velocity_projection_suppression_score
quality_guard_status
semantic_projection_status
adaptive_negative_fpr
adaptive_attack_success_status
adaptive_attack_claim_support_status
```

### 5.4 claim 降级

若 adaptive attack records 不存在或 checker 不通过, full_paper 只能声明:

```text
robustness_validated_under_non_adaptive_video_attacks
```

禁止声明:

```text
robust_to_flow_specific_adaptive_attacks
```

## 6. statistical_confidence_interval_reporter 接口规范

### 6.1 必须报告

```text
binomial_confidence_interval_for_fpr
binomial_confidence_interval_for_tpr
bootstrap_confidence_interval_for_tpr_at_fpr
cluster_by_video_confidence_interval
per_attack_family_confidence_interval
per_negative_family_confidence_interval
per_prompt_family_confidence_interval
```

### 6.2 输入记录

```text
records/event_scores.jsonl
records/thresholds.jsonl
records/baseline_scores.jsonl
records/ablation_scores.jsonl
manifests/full_paper_package_manifest.json
```

### 6.3 统计口径

报告必须同时保留:

```text
event_level_metric
unique_video_level_metric
cluster_by_video_metric
prompt_family_level_metric
attack_family_level_metric
negative_family_level_metric
```

若 event count 足够但 unique video count 不足, 必须输出统计可信度降级原因。

### 6.4 低 FPR 样本量阻断

当 `heldout_test_negative_event_count < 50000` 时, reporter 必须输出:

```text
low_fpr_validation_incomplete
sample_size_insufficient_for_fpr_0_001_claim
```

此时 `full_paper_result_checker` 不得允许 `TPR@FPR=0.001` 进入主 claim。

## 7. full_paper_result_checker 接口规范

### 7.1 输入

```text
records/event_scores.jsonl
records/trajectory_traces.jsonl
records/thresholds.jsonl
records/baseline_scores.jsonl
records/ablation_scores.jsonl
records/adaptive_attack_scores.jsonl
tables/main_detection_table.csv
tables/baseline_comparison_table.csv
tables/ablation_table.csv
figures/tpr_at_fpr_figure.json
figures/trajectory_evidence_figure.json
reports/statistical_confidence_interval_report.json
reports/claim_audit_report.json
reports/artifact_rebuild_report.json
manifests/full_paper_package_manifest.json
```

### 7.2 检查项

```text
threshold_source_split == calibration
test_time_threshold_update_blocked == true
target_fpr includes 0.001
heldout_clean_negative_fpr <= target_fpr
heldout_attacked_negative_fpr <= target_fpr
heldout_replay_negative_fpr <= target_fpr
heldout_sampler_mismatch_negative_fpr <= target_fpr
wrong_key_negative_fpr <= target_fpr
full_method_beats_modern_external_baseline
full_method_beats_internal_ablation_baseline
path_marginal_gain_at_fixed_fpr > 0
trajectory_payload_redundancy <= preset_limit
quality_degradation_within_limit == true
motion_degradation_within_limit == true
semantic_degradation_within_limit == true
confidence_interval_report_available == true
claim_audit_passed == true
artifact_rebuild_passed == true
```

### 7.3 输出

```text
artifacts/full_paper_result_decision.json
reports/full_paper_result_checker_report.md
```

决策字段:

```text
full_paper_result_checker_decision
full_paper_claim_allowed
blocking_requirement
observed_value
required_value
claim_downgrade_recommendation
submission_freeze_allowed
```

## 8. reviewer_evidence_index_builder 接口规范

### 8.1 输出

```text
reports/reviewer_evidence_index.json
reports/reviewer_evidence_index.md
```

### 8.2 每条 evidence index 记录

```text
reviewer_question_id
reviewer_question_category
paper_claim_id
required_evidence_artifact
supporting_record_path
supporting_table_path
supporting_figure_path
supporting_report_path
supporting_manifest_path
evidence_status
claim_downgrade_if_missing
```

### 8.3 必须覆盖的问题

```text
why_not_endpoint_only
why_not_post_hoc_video_watermark
why_not_explicit_temporal_alignment
why_flow_matching_specific
why_low_fpr_result_is_reliable
why_negative_tail_is_controlled
why_prompt_suite_is_not_cherry_picked
why_external_baselines_are_sufficient
why_ablation_is_sufficient
why_quality_degradation_is_acceptable
why_replay_or_sketch_evidence_is_trustworthy
why_results_are_reproducible
```

## 9. 验收测试要求

每个组件实现后, 至少需要轻量 functional tests 覆盖:

```text
missing_required_manifest_blocks_full_paper
test_split_threshold_update_blocks_full_paper
insufficient_negative_event_count_downgrades_fpr_0_001_claim
missing_modern_external_baseline_blocks_main_comparison
missing_ablation_variant_blocks_full_paper
missing_adaptive_attack_records_downgrades_adaptive_claim
missing_confidence_interval_report_blocks_submission_freeze
unsupported_claim_blocks_reviewer_evidence_index
```

这些测试必须使用 `tmp_path` 构造最小 records 和 manifests, 不得写入 checked-in `outputs/`。

## 10. Codex 实现顺序

推荐实现顺序:

```text
1. full_paper_dry_run_checker
2. statistical_confidence_interval_reporter
3. modern_external_baseline_runner 的 governed non-run record 与 adapter contract
4. reviewer_evidence_index_builder
5. full_paper_result_checker
6. flow_specific_adaptive_attack_runner
```

原因是前四项可以先用 validation-scale records 和 mock records 验证治理逻辑, 不需要立即启动大规模 GPU 实验。`flow_specific_adaptive_attack_runner` 成本较高, 应在 pilot 与 validation-scale 流程稳定后实现。

## 11. 不允许的捷径

以下行为必须视为阻断:

```text
手工填写 main_detection_table
用 pilot records 替代 full_paper records
用 test split 更新 threshold
用无法复现的外部论文数字作为主表 baseline
缺少 CI 时声明 TPR@FPR=0.001
adaptive attack 未完成时声明 Flow-specific robustness
reviewer evidence index 中引用不存在的 artifact
```

这些规则的目的不是增加流程复杂度, 而是防止项目在投稿阶段因为实验不充分、证据不可复现或 claim 过度而被拒稿。


## 12. external_baseline adapter comparison gate

full-paper dry-run 前必须区分两类 baseline 证据:

```text
external_baseline_status_records: 说明 baseline 是否登记、是否可运行、为何 non-run
external_baseline_comparison_records: 说明 adapter 是否在同一 run_root 上产出可重建 comparison 结果
```

当前已实现的 comparison 结果仍为 proxy control:

```text
explicit_dtw_temporal_alignment: measured_proxy
explicit_frame_matching_temporal_registration: measured_proxy
modern_video_watermark_baselines: unsupported_until_adapter_integrated
```

因此 full-paper checker 后续必须继续阻断以下行为:

```text
用 explicit synchronization proxy 代替现代视频水印 baseline 主表
用 unsupported modern baseline row 声明 SSTW 优于该 baseline
缺少 external_baseline_score_records.jsonl 时进入 baseline comparison claim
```


### 12.1 external_baseline 正式 claim 升级条件

external baseline comparison 从 proxy control 升级为 full-paper claim evidence 前, checker 必须同时看到以下证据:

```text
external_baseline_adapter_status: ready
external_baseline_output_record_status: governed_records_written
external_baseline_threshold_policy_compatible: true
external_baseline_attack_manifest_compatible: true
external_baseline_result_used_for_claim: true
metric_status: measured
claim_support_status: baseline_ready_for_main_comparison
```

若任一条件缺失, 该 baseline 只能进入 limitation / non-run / proxy comparison 说明, 不能进入主表正向 claim。
