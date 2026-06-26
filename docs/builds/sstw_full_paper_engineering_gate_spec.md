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
让 Codex 在后续工程实现时, 能按固定接口逐步实现 validation_scale gate、pilot_paper gate、full_paper_result_checker、modern baseline runner、adaptive attack runner、statistical CI reporter、reviewer evidence index builder 和轻量判定组件。
```

## 1. 总体工程门禁架构

paper 级主干门禁只保留以下顺序:

```text
protocol_governance -> mechanism_validation -> validation_scale -> pilot_paper -> full_paper -> submission_freeze
```

`small_scale_claim_pilot_gate` 不再作为主干门禁使用, 只能作为 `mechanism_validation` 下的历史小样本机制检查记录。`generative_video_model_probe` 不再作为独立门禁使用, 只表示真实生成式视频模型实验的实现包或 phase 文档。正式阶段跳转必须以主干门禁和本节列出的轻量判定为准。

paper 级工程组件分为核心门禁、支撑 runner / reporter 和轻量判定三类:

| 组件 | 作用 | 产物性质 |
|---|---|---|
| `validation_scale_readiness_gate` | 进入 paper 级运行前的小样本全流程打通门禁, 以 FPR=0.10 级别验证完整论文产物链路是否可生成 | gate decision |
| `pilot_paper_gate` | 在 validation_scale 通过后小规模跑完整 paper 协议并产出 pilot 级论文结果 | gate decision |
| `modern_external_baseline_runner` | 在项目内完成 external baseline 的 clone / build / run / adapt / record, 产出 measured_formal 或受治理 non-run 记录 | governed records + lightweight decision |
| `flow_specific_adaptive_attack_runner` | 运行 Flow-specific adaptive attacks | governed records |
| `statistical_confidence_interval_reporter` | 计算低 FPR、TPR 和 cluster-by-video 置信区间, 并输出是否可支撑 claim 的 decision artifact | governed reports + lightweight decision |
| `full_paper_result_checker` | 检查 full_paper records、tables、figures、reports 是否满足论文主张 | gate decision |
| `reviewer_evidence_index_builder` | 将审稿问题映射到 governed artifacts, 由 submission_freeze 消费 | governed report |
| `stage_transition_decision` | 生成阶段明确的跳转判定, 包括 validation_scale -> pilot_paper、pilot_paper -> full_paper、full_paper -> submission_freeze; 该判定只能在 source gate 已 PASS 后生成, 不得作为 source gate 自身的 PASS 前置 | lightweight decision |
| `external_baseline_self_containment_decision` | 检查 modern external baseline 是否全部由项目内自包含流程产出, non-run record 只能阻断不能替代 measured_formal | lightweight decision |
| `data_split_and_leakage_guard` | 检查 calibration / held-out test / stress / ablation split 隔离、video identity 泄漏和 threshold 来源 | lightweight decision |

这些组件属于通用科研工程门禁。SSTW 的项目特定要求在于: 所有组件必须围绕 Flow Matching 轨迹水印的 state-space evidence、path evidence、endpoint evidence、replay uncertainty 和 fixed low-FPR calibration 展开。

## 2. 推荐实现位置

推荐实现路径如下:

```text
experiments/generative_video_model_probe/pilot_paper_gate.py
scripts/check_results/full_paper_result_checker.py
experiments/generative_video_model_probe/external_baseline_runner.py
experiments/flow_specific_adaptive_attack/runner.py
experiments/flow_specific_adaptive_attack/table_builder.py
experiments/flow_specific_adaptive_attack/package_outputs.py
main/analysis/statistical_confidence_intervals.py
experiments/submission_freeze_preparation/reviewer_evidence_index.py
scripts/check_results/stage_transition_decision.py
scripts/check_results/external_baseline_self_containment_decision.py
scripts/check_results/data_split_and_leakage_guard.py
tests/functional/test_pilot_paper_gate.py
tests/functional/test_full_paper_result_checker.py
tests/functional/test_statistical_confidence_intervals.py
tests/functional/test_reviewer_evidence_index.py
tests/functional/test_lightweight_gate_decisions.py
```

若后续实现中选择其他路径, 必须保持语义名称清晰, 并同步更新阶段文档和 harness 可见的路径规则。

## 3. validation_scale_readiness_gate 接口规范

`validation_scale` 的正式定义是“小样本全流程打通验证”。它不是 full_paper 效果证明, 也不是只检查单个 runner 是否能启动的 smoke test。该 gate 的功能是作为 paper 级前的全流程打通层, 在 `FPR=0.10` 小样本口径下提前验证 `pilot_paper` 和 `full_paper` 会需要的产物类型是否都能由本项目自动生成。

### 3.1 输入

必须输入:

```text
generation_records
formal_quality_motion_semantic_records
runtime_attack_records
runtime_detection_records
external_baseline_records
external_baseline_score_records
validation_internal_ablation_records
adaptive_attack_records
statistical_confidence_interval_decision
validation_artifact_rebuild_dry_run_decision
method_mechanism_validation_decision
external_baseline_self_containment_decision
data_split_and_leakage_guard_decision
motion_threshold_calibration_decision
```

`validation_scale_readiness_gate` 不读取 `validation_scale_to_pilot_paper_transition_decision`, 因为该跳转判定必须在 `validation_scale_gate_decision == PASS` 之后才能生成。这样可以避免“validation_scale PASS 依赖跳转判定, 跳转判定又依赖 validation_scale PASS”的循环依赖。

### 3.2 输出

必须输出:

```text
records/validation_scale_gate_records.jsonl
tables/validation_scale_gate_table.csv
figures/validation_scale_gate_figure.json
artifacts/validation_scale_gate_decision.json
reports/validation_scale_gate_report.md
manifests/validation_scale_package_manifest.json
```

其中 `figures/validation_scale_gate_figure.json` 可以是最小诊断图 manifest, 但必须由 records 重建, 不能手工填写。若当前实现暂未生成该图, checker 必须把它列为操作手册执行闭环缺口。

### 3.3 PASS 条件

只有同时满足以下条件, `validation_scale_readiness_gate` 才能写出 PASS。进入 `pilot_paper` 还必须在该 PASS 落盘后生成 `validation_scale_to_pilot_paper_transition_decision`; 该 PASS 不直接允许 full_paper claim:

```text
target_fpr == 0.10
all_required_phase_decisions_exist
all_required_phase_decisions_passed_or_explicitly_downgraded
external_baseline_self_containment_decision_passed
data_split_and_leakage_guard_passed
external_baseline_self_contained_outputs_ready
modern_external_baseline_formal_measured_adapter_count >= 6
external_baseline_measured_adapter_count >= 8
internal_ablation_records_ready
adaptive_attack_records_ready
replay_or_sketch_records_ready_or_claim3_downgraded
confidence_interval_report_ready
artifact_rebuild_dry_run_ready
claim_audit_or_claim_boundary_report_ready
package_manifest_ready
```

`validation_scale` 通过后仍不得声明 `TPR@FPR=0.001` 或 full_paper 主表结论。它只能说明 paper 级结果生产流程已经小样本跑通。它是进入 `pilot_paper` 和后续 `full_paper` 流程的必要条件, 但不是进入 `full_paper` 的充分条件; `full_paper` 仍必须等待 `pilot_paper_gate`、`full_paper_result_checker`、CI、claim audit 和 artifact rebuild 全部通过。

`validation_scale_gate_decision == PASS` 落盘后, 才允许运行 `stage_transition_decision` 生成 `validation_scale_to_pilot_paper_transition_decision`。该跳转判定只决定是否允许进入 `pilot_paper`, 不反向参与 `validation_scale` 自身 PASS 计算。

## 4. pilot_paper_gate 接口规范

### 4.1 输入

`pilot_paper_gate` 读取已经落盘的 pilot_paper records 与前置 decision artifacts。该 gate 允许读取 pilot_paper 的 held-out test split, 因为它本身就是小规模完整协议执行阶段; 但它不得读取 full_paper held-out test 结果, 也不得把 pilot_paper 结果外推为 full_paper 主表。

必须输入:

```text
phase_decision_records
prompt_suite_manifest
seed_plan_manifest
generation_records
formal_quality_motion_semantic_records
runtime_detection_records
method_mechanism_validation_records
motion_threshold_calibration_decision
method_mechanism_validation_decision
validation_scale_gate_decision
validation_scale_to_pilot_paper_transition_decision
```

可选输入:

```text
previous_validation_summary
external_baseline_comparison_summary
adaptive_attack_summary
replay_or_sketch_gate_summary
```

### 4.2 禁止输入

```text
full_paper_heldout_test_event_scores
full_paper_heldout_test_detection_scores
manual_table_values
manually_selected_best_threshold
```

这些输入若出现, checker 必须失败, 因为 pilot_paper 只能使用自己的 calibration/test split, 不能接触 full_paper test split, 也不能手工选择最佳阈值。

### 4.3 输出

必须输出:

```text
artifacts/pilot_paper_gate_decision.json
thresholds/pilot_paper_frozen_threshold.json
records/pilot_paper_gate_records.jsonl
tables/pilot_paper_gate_table.csv
reports/pilot_paper_gate_report.md
```

`pilot_paper_gate_decision.json` 必须至少包含:

```text
pilot_paper_gate_decision
pilot_paper_claim_allowed
full_paper_allowed
missing_pilot_paper_requirements
next_allowed_action
next_forbidden_action
claim_support_status
validation_scale_gate_decision
threshold_source_split
test_time_threshold_update_blocked
calibration_negative_event_count
heldout_test_negative_event_count
heldout_attacked_positive_event_count
tpr_at_fpr_01
tpr_at_fpr_001_claim_allowed
```

### 4.4 PASS 条件

只有同时满足以下条件, 才能输出 `pilot_paper_claim_allowed = true`。即使该 gate 通过, `full_paper_allowed` 仍必须为 `false`, 因为 pilot_paper 只产出 pilot 级论文结果:

```text
all_required_phase_decisions_exist
all_required_phase_decisions_passed_or_downgraded
method_mechanism_validation_passed
validation_scale_gate_passed
validation_scale_to_pilot_paper_transition_decision_passed
motion_threshold_calibration_ready
formal_motion_claim_ready
pilot_paper_profile_generation_records_ready
pilot_paper_calibration_split_ready
pilot_paper_heldout_test_split_ready
calibration_negative_event_count_ready
heldout_test_negative_event_count_ready
heldout_attacked_positive_event_count_ready
frozen_threshold_artifact_computable
heldout_fpr_within_target
tpr_at_fpr_01_computable
path_marginal_gain_ready
negative_tail_not_inflated
wrong_sampler_replay_rejected
```

## 5. modern_external_baseline_runner 接口规范

### 5.1 baseline 分层

runner 必须支持至少三种 baseline 状态:

```text
runnable
non_runnable_with_governed_reason
protocol_incompatible_with_governed_reason
```

不得静默删除现代 baseline。无法运行时也必须写出 governed record。

现代 baseline 的正式运行路径必须是项目内自包含产出:

```text
project_clone
project_build
project_run
project_adapt
project_record
```

不允许把外部补交 result bundle、手写 JSON、NPZ 分数文件、论文表格数字或 SSTW proxy 分数转换为主表 baseline 结果。若官方资源无法在项目流程内获得, runner 必须写出 `non_runnable_with_governed_reason`。

### 5.2 记录字段

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
external_baseline_project_clone_status
external_baseline_project_build_status
external_baseline_project_run_status
external_baseline_project_adapt_status
external_baseline_project_record_status
```

其中, `external_baseline_result_used_for_claim` 只有在 baseline 真实运行且协议兼容时才允许为 true。

### 5.3 主表准入

进入主表的 baseline 必须满足:

```text
external_baseline_runnable_status == runnable
external_baseline_adapter_status == ready
external_baseline_output_record_status == governed_records_written
external_baseline_project_clone_status == completed
external_baseline_project_build_status == completed
external_baseline_project_run_status == completed
external_baseline_project_adapt_status == completed
external_baseline_project_record_status == completed
metric_status == measured_formal
external_baseline_threshold_policy_compatible == true
external_baseline_attack_manifest_compatible == true
external_baseline_result_used_for_claim == true
```

若只引用外部论文数字, 只能进入 related work 或 supplementary discussion, 不能进入主表胜负判断。若 baseline 只能写出 `non_runnable_with_governed_reason`, 该记录只能解释阻断原因, 不能替代正式 `measured_formal` 主表结果。

### 5.4 external_baseline_self_containment_decision 输出

runner 或 checker 必须额外写出轻量判定:

```text
artifacts/external_baseline_self_containment_decision.json
reports/external_baseline_self_containment_report.md
```

该判定必须逐 baseline 检查 `project_clone`、`project_build`、`project_run`、`project_adapt` 和 `project_record`。只有全部进入主表的现代 baseline 同时满足 `metric_status == measured_formal` 和项目内自包含产出链路, 该 decision 才允许通过。

## 6. flow_specific_adaptive_attack_runner 接口规范

### 6.1 攻击者知识层级

runner 至少应覆盖:

```text
black_box_video_only_attacker
gray_box_sampler_signature_attacker
white_box_oracle_limited_flow_attacker
```

### 6.2 攻击目标

至少应覆盖:

```text
endpoint_preserving_path_perturbation
path_response_cancellation
velocity_projection_suppression
time_grid_or_scheduler_mismatch
trajectory_sketch_replacement
public_negative_tail_probe
```

### 6.3 必须记录字段

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

### 6.4 claim 降级

若 adaptive attack records 不存在或 checker 不通过, full_paper 只能声明:

```text
robustness_validated_under_non_adaptive_video_attacks
```

禁止声明:

```text
robust_to_flow_specific_adaptive_attacks
```

## 7. statistical_confidence_interval_reporter 接口规范

### 7.1 必须报告

```text
binomial_confidence_interval_for_fpr
binomial_confidence_interval_for_tpr
bootstrap_confidence_interval_for_tpr_at_fpr
cluster_by_video_confidence_interval
per_attack_family_confidence_interval
per_negative_family_confidence_interval
per_prompt_family_confidence_interval
```

### 7.2 输出产物

```text
records/statistical_confidence_interval_records.jsonl
tables/statistical_confidence_interval_table.csv
artifacts/statistical_confidence_interval_decision.json
reports/statistical_confidence_interval_report.md
```

`reports/statistical_confidence_interval_report.md` 是统计说明, `artifacts/statistical_confidence_interval_decision.json` 是后续 `full_paper_result_checker` 和 `submission_package_freeze` 消费的轻量阻断判定。二者必须同时存在, 避免出现“报告存在但是否允许 claim 不清楚”的状态。

### 7.3 输入记录

```text
records/event_scores.jsonl
records/thresholds.jsonl
records/baseline_scores.jsonl
records/ablation_scores.jsonl
manifests/full_paper_package_manifest.json
```

### 7.4 统计口径

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

### 7.5 低 FPR 样本量阻断

当 `heldout_test_negative_event_count < 50000` 时, reporter 必须输出:

```text
low_fpr_validation_incomplete
sample_size_insufficient_for_fpr_0_001_claim
```

此时 `full_paper_result_checker` 不得允许 `TPR@FPR=0.001` 进入主 claim。

## 8. full_paper_result_checker 接口规范

`full_paper_result_checker` 必须读取 `configs/protocol/full_paper_generative_probe.json` 作为正式协议来源。若运行目录、Notebook 参数或 manifest 与该配置冲突, checker 必须以配置为准并 fail closed。

### 8.1 输入

```text
configs/protocol/full_paper_generative_probe.json
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
artifacts/statistical_confidence_interval_decision.json
artifacts/pilot_paper_to_full_paper_transition_decision.json
reports/claim_audit_report.json
reports/artifact_rebuild_report.json
manifests/full_paper_package_manifest.json
```

### 8.2 检查项

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
statistical_confidence_interval_decision_passed == true
claim_audit_passed == true
artifact_rebuild_passed == true
```

### 8.3 输出

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

## 9. reviewer_evidence_index_builder 接口规范

`reviewer_evidence_index_builder` 必须在 `full_paper_result_checker_decision == PASS` 或明确的 downgrade decision 落盘后运行。它只索引已经由 checker 确认可使用的 claims、tables、figures、reports 和 manifests, 不能把尚未通过的 full_paper 结果提前包装成审稿证据。

### 9.1 输出

```text
reports/reviewer_evidence_index.json
reports/reviewer_evidence_index.md
```

### 9.2 每条 evidence index 记录

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

### 9.3 必须覆盖的问题

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


## 10. 三个轻量判定接口规范

轻量判定不新增重型实验阶段, 只负责在阶段跳转或证据边界升级时 fail closed。它们必须写出 machine-readable artifact, 并被下游 gate 消费。`stage_transition_decision` 有一个额外约束: 它只能被 target gate 消费, 不能被 source gate 消费, 否则会产生循环依赖。

### 10.1 stage_transition_decision

```text
artifacts/stage_transition_decision.json
reports/stage_transition_report.md
```

正式实现时必须同时写出阶段明确的别名 artifact, 避免一个泛化文件在不同跳转中被误用:

```text
artifacts/validation_scale_to_pilot_paper_transition_decision.json
artifacts/pilot_paper_to_full_paper_transition_decision.json
artifacts/full_paper_to_submission_freeze_transition_decision.json
```

必须检查:

```text
source_stage
target_stage
source_gate_decision == PASS
forbidden_skip_detected == false
target_profile_allowed == true
required_protocol_config_present == true
blocking_requirements == []
```

执行顺序必须是:

```text
source gate 写出 PASS decision
-> stage_transition_decision 检查 source_stage / target_stage / protocol config
-> target gate 读取对应 transition decision
```

因此:

```text
validation_scale_readiness_gate 不读取 validation_scale_to_pilot_paper_transition_decision
pilot_paper_gate 读取 validation_scale_to_pilot_paper_transition_decision
full_paper_result_checker 读取 pilot_paper_to_full_paper_transition_decision
submission_package_freeze 读取 full_paper_to_submission_freeze_transition_decision
```

该判定用于防止从 `method_mechanism_validation`、历史 small-scale pilot 记录或 `validation_scale` 直接跳到不合法阶段。`validation_scale` 通过只允许进入 `pilot_paper` 或继续准备 `full_paper`; `full_paper` 的正式 claim 仍由 `full_paper_result_checker` 决定。

### 10.2 external_baseline_self_containment_decision

```text
artifacts/external_baseline_self_containment_decision.json
reports/external_baseline_self_containment_report.md
```

必须检查进入主表候选的现代 baseline 均满足:

```text
external_baseline_project_clone_status == completed
external_baseline_project_build_status == completed
external_baseline_project_run_status == completed
external_baseline_project_adapt_status == completed
external_baseline_project_record_status == completed
metric_status == measured_formal
external_baseline_result_used_for_claim == true
```

若某个 baseline 只有 governed non-run record, 该 decision 必须失败或降级, 不能把 non-run record 当作 measured baseline。

### 10.3 data_split_and_leakage_guard

```text
artifacts/data_split_and_leakage_guard_decision.json
reports/data_split_and_leakage_guard_report.md
```

必须检查:

```text
calibration_split_disjoint_from_heldout_test == true
threshold_source_split == calibration
test_time_threshold_update_blocked == true
video_identity_leakage_detected == false
prompt_seed_leakage_detected == false
cluster_by_video_manifest_available == true
negative_family_counts_match_protocol == true
```

该判定属于通用工程写法, 目的是把 split 泄漏、阈值泄漏和 video identity 泄漏提前暴露, 不承担替代统计 CI 或 full_paper_result_checker 的职责。

## 11. 验收测试要求

每个组件实现后, 至少需要轻量 functional tests 覆盖:

```text
missing_required_manifest_blocks_full_paper
test_split_threshold_update_blocks_full_paper
insufficient_negative_event_count_downgrades_fpr_0_001_claim
missing_modern_external_baseline_blocks_main_comparison
missing_ablation_variant_blocks_full_paper
missing_adaptive_attack_records_downgrades_adaptive_claim
missing_confidence_interval_report_blocks_submission_freeze
missing_stage_transition_decision_blocks_stage_jump
missing_external_baseline_self_containment_blocks_main_comparison
split_leakage_blocks_full_paper_claim
unsupported_claim_blocks_reviewer_evidence_index
```

这些测试必须使用 `tmp_path` 构造最小 records 和 manifests, 不得写入 checked-in `outputs/`。

## 12. Codex 实现顺序

推荐实现顺序:

```text
1. protocol_governance 与 mechanism_validation 基础检查
2. modern_external_baseline_runner 的项目内 clone / build / run / adapt / record 与 measured_formal records
3. external_baseline_self_containment_decision 和 data_split_and_leakage_guard
4. internal_ablation_matrix_runner
5. flow_specific_adaptive_attack_runner
6. replay_and_authenticated_sketch_gate 或受治理 Claim-3 downgrade gate
7. statistical_confidence_interval_reporter
8. validation_scale_readiness_gate
9. stage_transition_decision, 先生成 validation_scale_to_pilot_paper_transition_decision
10. pilot_paper_gate
11. pilot_paper_to_full_paper_transition_decision
12. full_paper_result_checker
13. reviewer_evidence_index_builder
14. full_paper_to_submission_freeze_transition_decision
15. submission_package_freeze
```

原因是 `validation_scale` 是进入 paper 级运行前的小样本全流程打通层。所有 paper 相关机制必须先在 FPR=0.10 小样本口径下产出 governed records、tables、figures、reports、manifests 和 claim audit, 再由 `stage_transition_decision` 生成 `validation_scale_to_pilot_paper_transition_decision`, 之后才能进入 `pilot_paper` 或继续准备 `full_paper` 后续流程; `full_paper` claim 仍由 `full_paper_result_checker` 放行。如果 external baseline、内部消融、adaptive attack、replay/sketch、CI 或 artifact rebuild 在 validation_scale 阻断, 后续步骤不得用 paper 级运行补造缺失产物。`reviewer_evidence_index_builder` 必须晚于 `full_paper_result_checker`, 因为它只能索引已经由 checker 确认的 claims、tables、figures、reports 和 manifests, 不能在 claim 未通过前提前构造审稿证据。

## 13. 不允许的捷径

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


## 14. external_baseline adapter comparison gate

pilot_paper 前必须区分两类 baseline 证据:

```text
external_baseline_status_records: 说明 baseline 是否登记、是否可运行、为何 non-run
external_baseline_comparison_records: 说明 adapter 是否在同一 run_root 上通过项目内 clone / build / run / adapt / record 产出可重建 comparison 结果
external_baseline_source_intake_manifest: 说明第三方 source、adapter 和官方命令配置边界
external_baseline_execution_manifest: 说明本次 baseline comparison 的 execution boundary、formal rows 和 evidence paths
```

validation_scale 之前必须能够区分三类结果:

```text
explicit_dtw_temporal_alignment: measured_proxy
explicit_frame_matching_temporal_registration: measured_proxy
modern_video_watermark_baselines: measured_formal 或 unsupported_with_reason
```

因此 full_paper_result_checker 后续必须继续阻断以下行为:

```text
用 explicit synchronization proxy 代替现代视频水印 baseline 主表
用 unsupported modern baseline row 声明 SSTW 优于该 baseline
缺少 external_baseline_score_records.jsonl 时进入 baseline comparison claim
缺少 external_baseline_execution_manifest.json 时进入 baseline comparison claim
缺少 source intake / source inspection / clone / build / run / adapt / record 清单时进入 validation_scale PASS
```


### 13.1 external_baseline 正式 claim 升级条件

external baseline comparison 从 proxy control 升级为 full_paper claim evidence 前, checker 必须同时看到以下证据:

```text
external_baseline_adapter_status: ready
external_baseline_output_record_status: governed_records_written
external_baseline_threshold_policy_compatible: true
external_baseline_attack_manifest_compatible: true
external_baseline_result_used_for_claim: true
metric_status: measured_formal
modern_external_baseline_formal_measured_adapter_count >= 6
missing_modern_external_baseline_formal_adapter_names == []
external_baseline_execution_manifest.json: present
external_baseline_project_clone_status: completed
external_baseline_project_build_status: completed
external_baseline_project_run_status: completed
external_baseline_project_adapt_status: completed
external_baseline_project_record_status: completed
claim_support_status: baseline_ready_for_main_comparison
```

若任一条件缺失, 该 baseline 只能进入 limitation / non-run / proxy comparison 说明, 不能进入主表正向 claim。
