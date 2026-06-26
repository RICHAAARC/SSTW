# submission_package_freeze 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段将 governed records 转换为论文可使用的 tables、figures、reports 和 manifests, 并执行 claim audit、readiness summary 和 release extraction 检查。

### 1.2 输入

```text
records/event_scores.jsonl
records/thresholds.jsonl
records/trajectory_traces.jsonl
experiments/submission_freeze_preparation/
scripts/package_results/submission_freeze_preparation_packager.py
docs/builds/sstw_project_construction_flow.md
docs/builds/sstw_method_mechanism_design.md
```

### 1.3 构建任务

1. 从 records 重建主表、baseline 表和 ablation 表。
2. 从 records 和 manifests 重建主图所需数据。
3. 生成 claim audit report。
4. 生成 readiness summary。
5. 生成 submission package manifest。
6. 执行 release extraction contract 检查。

### 1.4 必须产物

```text
tables/main_detection_table.csv
tables/baseline_comparison_table.csv
tables/ablation_table.csv
figures/roc_or_tpr_at_fpr_figure.json
figures/trajectory_evidence_figure.json
reports/claim_audit_report.json
reports/readiness_summary.json
manifests/submission_package_manifest.json
```

### 1.5 禁止事项

1. 不得手工改写正式表格数值。
2. 不得从 test split 反向更新 calibration threshold。
3. 不得用 placeholder 字段支撑 supported claims。
4. 不得把临时 Colab 输出直接当成论文 artifact。

### 1.6 通过标准

1. 主表、主图、报告和 claim audit 可由 records 与 manifests 自动重建。
2. `pytest -q` 和 harness 审计通过。
3. supported claims 全部绑定 governed artifacts。

### 1.7 审稿风险包

submission package 除主结果外, 还必须包含审稿风险回应材料:

```text
reports/baseline_sufficiency_report.md
reports/ablation_sufficiency_report.md
reports/adaptive_attack_report.md
reports/low_fpr_confidence_interval_report.md
reports/prompt_observability_audit_report.md
reports/replay_and_sketch_evidence_report.md
reports/artifact_rebuild_report.md
```

这些报告不应手写结论, 而应由 records、tables 和 manifests 重建。

### 1.8 不允许冻结的情况

```text
pilot_paper_gate_decision != PASS
full_paper_result_checker_decision != PASS
claim_audit_passed != true
artifact_rebuild_passed != true
modern_external_baseline_records_missing = true
external_baseline_self_containment_decision != PASS
full_paper_to_submission_freeze_transition_decision != PASS
data_split_and_leakage_guard_decision != PASS
adaptive_attack_records_missing = true
low_fpr_confidence_interval_missing = true
statistical_confidence_interval_decision != PASS
```

### 1.9 审稿证据索引

submission package 必须生成 reviewer evidence index, 用于把潜在审稿问题映射到 records、tables、figures、reports 和 manifests:

```text
reviewer_question_id
reviewer_question_category
paper_claim_id
supporting_record_path
supporting_table_path
supporting_figure_path
supporting_report_path
supporting_manifest_path
evidence_status
claim_downgrade_if_missing
```

该 index 必须至少覆盖:

```text
why_not_endpoint_only
why_not_post_hoc_video_watermark
why_not_explicit_temporal_alignment
why_flow_matching_specific
why_low_fpr_result_is_reliable
why_external_baselines_are_sufficient
why_ablation_is_sufficient
why_results_are_reproducible
```

若 evidence index 不能重建或存在 unsupported claim, 本阶段不得给出 ready-for-submission 结论。

### 1.10 reviewer evidence index 工程规范索引

reviewer evidence index 的字段、覆盖问题和阻断规则必须遵守:

```text
docs/builds/sstw_full_paper_engineering_gate_spec.md
```

该 index 不是人工 rebuttal 文案, 而是 governed artifacts 的索引。若索引中的任一 supporting path 不存在或不可由 manifest 追溯, 对应 claim 必须降级。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有 submission freeze preparation 相关模块:

```text
experiments/submission_freeze_preparation/runner.py
experiments/submission_freeze_preparation/main_tables.py
experiments/submission_freeze_preparation/readiness_summary.py
scripts/package_results/submission_freeze_preparation_packager.py
```

### 2.2 当前阶段使用边界

该阶段只能组织和重建 governed artifacts, 不能手工创造论文结果。若上游阶段缺少真实 GPU、真实模型 records、pilot gate 记录或 negative family 记录, 本阶段只能报告 evidence gap, 不能补写 supported claims。

最新 small-scale pilot 已通过, 且现代外部 baseline 已有 governed status / non-run records。该状态只能说明 submission freeze 的部分上游材料开始具备可追溯入口, 不能说明 submission package 可以冻结; non-run records 只能作为阻断说明, 不能替代 `measured_formal` external baseline。当前仍缺少 pilot_paper 真实 GPU 结果、validation_scale、full_paper 主表 records、现代 baseline 主表对比 records、内部消融、adaptive attack、replay/sketch 和低 FPR 统计报告。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 结构就绪, 未进入最终冻结 |
| 主要差距项 | small-scale pilot 已通过, 但上游 validation_scale、pilot_paper、full_paper records 与三个轻量判定不存在或未通过, 只能报告 evidence gap。 |
| 下一步构建方向 | 等待 validation_scale、pilot_paper 与 full_paper_result_checker 通过后, 再重建 tables、figures、reports、reviewer evidence index 和 claim audit。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 结构就绪, 未进入最终冻结
gap_item: small-scale pilot 已通过, 但上游 validation_scale、pilot_paper、full_paper records 与三个轻量判定不存在或未通过, 只能报告 evidence gap。
next_action: 等待 validation_scale、pilot_paper 与 full_paper_result_checker 通过后, 再重建 tables、figures、reports、reviewer evidence index 和 claim audit。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```

### 3.2 2026-06-23 最新冻结边界

当前禁止直接进入 submission freeze:

```text
method_mechanism_validation_passed = true
pilot_paper_result_records_ready = false
external_baseline_status_records_ready = true
validation_scale_full_pipeline_completed = false
full_paper_result_records_ready = false
claim_audit_for_full_paper_passed = false
artifact_rebuild_for_full_paper_passed = false
submission_freeze_allowed = false
```

可继续推进的工程工作是 submission freeze 的 checker、reviewer evidence index builder 和 artifact rebuild dry-run 接口; 不可推进的工作是生成最终论文主表、最终 claim audit 或 ready-for-submission 结论。
