# SSTW 顶刊顶会实验充分性核查清单

## 0. 文档定位

本文档用于独立核查 SSTW 项目是否具备面向顶刊顶会投稿的实验充分性。它不记录当前项目完成进度, 也不替代 `sstw_project_construction_flow.md` 的总体构建顺序。

本文档回答的问题是:

```text
如果 Codex 按项目手册推进, 是否能尽量避免因为实验不足、baseline 不足、消融不足、低 FPR 证据不足或复现证据不足而被拒稿。
```

当前进度应记录在:

```text
docs/builds/sstw_phase_completion_status.md
docs/builds/phases/
```

## 1. 顶刊顶会实验最低充分性

顶刊顶会版本至少需要同时满足:

```text
mechanism_evidence_sufficient
real_model_evidence_sufficient
low_fpr_evidence_sufficient
external_baseline_evidence_sufficient
internal_ablation_evidence_sufficient
adaptive_attack_evidence_sufficient
quality_and_utility_evidence_sufficient
replay_or_claim3_downgrade_evidence_sufficient
artifact_rebuild_evidence_sufficient
reproducibility_evidence_sufficient
```

若任一项缺失, 论文可以继续作为 validation 或 workshop 版本推进, 但不应直接生成 full_paper 结果包。

## 2. 机制证据核查

| 核查项 | 必须回答的问题 | 必须证据 |
|---|---|---|
| Flow Matching 内生性 | 水印是否真正进入采样轨迹, 而不是后处理视频 | velocity constraint records、trajectory trace records |
| endpoint 一致性 | 速度场弱约束是否与 endpoint evidence 一致 | endpoint consistency records |
| path 独立性 | path evidence 是否不是 endpoint evidence 的重复 | path marginal gain table、redundancy audit |
| state posterior 必要性 | 状态空间后验是否优于普通聚合器 | ablation table、generic SSM control |
| admissibility 必要性 | 状态搜索约束是否降低 negative tail | negative tail audit |
| fixed-FPR 合规 | 阈值是否只来自 calibration split | thresholds records、threshold audit |

## 3. 数据集与 prompt 充分性核查

数据集必须证明结果不是 prompt cherry-picking 或弱运动样本偶然造成。必须检查:

```text
prompt_suite_manifest_frozen
prompt_family_balanced
motion_pattern_balanced
foreground_scale_requirement_recorded
expected_motion_observability_recorded
prompt_observability_audit_passed
old_run_records_preserved
no_detection_score_used_for_prompt_filtering
```

如果 prompt 在 pilot 中被修复, 必须保留旧 run 的失败记录, 并说明修复只影响未来 run。

## 4. 低 FPR 统计充分性核查

`TPR@FPR=0.001` 不能只由少量 negative 样本支持。full_paper 前必须检查:

```text
calibration_negative_event_count >= 50000
heldout_test_negative_event_count >= 50000
negative_event_count_per_family >= 5000
threshold_source_split == calibration
test_time_threshold_update_blocked == true
binomial_confidence_interval_for_fpr_available
bootstrap_confidence_interval_for_tpr_at_fpr_available
cluster_by_video_confidence_interval_available
```

若 negative event 数量不足, 只能写为:

```text
sample_size_insufficient_for_fpr_0_001_claim
```

不得将其写成 full_paper 主结论。

## 5. 外部 baseline 充分性核查

外部 baseline 至少覆盖:

```text
in_generation_or_diffusion_video_watermark_baseline
post_hoc_neural_video_watermark_baseline
explicit_temporal_alignment_control
endpoint_only_control
generic_state_space_or_temporal_aggregator_control
```

每个现代 baseline 必须有:

```text
external_baseline_name
external_baseline_source_url
external_baseline_runnable_status
external_baseline_adapter_status
external_baseline_protocol_gap
external_baseline_output_record_status
external_baseline_result_used_for_claim
```

如果只比较 image watermark、frame watermark 或 endpoint-only control, 则 baseline 充分性不通过。

## 6. 内部消融充分性核查

内部消融必须覆盖以下问题:

```text
without_velocity_field_weak_constraint 是否下降
endpoint_only_control 是否不足
trajectory_only_control 是否不足
without_path_invariant_observation 是否下降
without_replay_uncertainty 是否在 replay 设置下降
without_admissibility 是否抬高 negative tail
key_agnostic_state_space 是否下降
generic_state_space 是否下降
explicit_temporal_alignment 是否不足
without_quality_guard 是否造成质量退化
```

每个消融必须在相同 split、相同 attack manifest 和相同 threshold policy 下比较。

## 7. 攻击与鲁棒性充分性核查

攻击至少覆盖:

```text
spatial_transform_attack
temporal_resampling_attack
compression_attack
frame_drop_or_duplication_attack
generative_recompression_attack
endpoint_preserving_path_perturbation_attack
flow_time_grid_mismatch_attack
wrong_sampler_replay_attack
wrong_prompt_replay_attack
wrong_key_attack
```

如果 flow-specific adaptive attack 未完成, 必须降级 adaptive robustness claim, 不能把普通视频攻击结果写成 Flow-specific robustness。

## 8. 质量与效用充分性核查

SSTW 不能只报告检测率。必须同时报告:

```text
visual_quality_metric
semantic_consistency_metric
motion_consistency_metric
temporal_consistency_metric
generation_overhead
detection_overhead
attack_runtime_overhead
```

若水印强度提升导致质量显著下降, 需要报告 trade-off 曲线, 不能只选择一个有利强度点。

## 9. artifact rebuild 与复现充分性核查

所有论文结果必须能从 records 和 manifests 重建。必须检查:

```text
records_schema_audit_passed
threshold_audit_passed
baseline_records_audit_passed
ablation_records_audit_passed
claim_audit_passed
artifact_rebuild_passed
package_manifest_complete
code_version_recorded
dependency_lock_recorded
run_command_recorded
```

禁止把 Notebook 中的临时变量、手工表格、截图或未登记外部数值作为论文主证据。

## 10. 常见拒稿风险与阻断规则

| 风险 | 阻断规则 |
|---|---|
| 只有 pilot, 没有 full validation | 不允许 full_paper |
| 只有内部 baseline, 没有现代外部 baseline | 不允许 full_paper |
| 只有 TPR@FPR=0.01, 没有 TPR@FPR=0.001 | 降级低 FPR claim |
| 只有 event count, 没有 unique video count | 降级统计可信度 claim |
| 只有后处理攻击, 没有 Flow-specific adaptive attack | 降级 robustness claim |
| replay/sketch 未闭合 | 降级 Claim-3 |
| 表格不可重建 | 不允许 submission freeze |
| claim audit 失败 | 不允许 submission freeze |

## 11. Codex 执行检查清单

Codex 每次推进到下一阶段前, 应回答:

```text
current_blocking_gate 是什么
本轮是否只做文档修改
是否生成 governed records
是否生成或修改 full_paper artifacts
pytest 是否通过
harness 是否通过
是否存在 checked-in outputs
是否存在 placeholder 支撑 claim
是否存在 test split threshold update
下一步允许执行什么
下一步禁止执行什么
```

若无法回答上述问题, 不应推进阶段状态。

## 12. 工程化 readiness 评分

顶会实验充分性不只取决于手册是否完整, 还取决于手册中的 gate 是否已经工程化。建议使用以下评分:

| 项目 | 分值 |
|---|---:|
| full_paper dry-run checker 已实现并测试通过 | 15 |
| modern external baseline runner 或 governed non-run record 已实现 | 15 |
| internal ablation matrix 已实现并能重建表格 | 10 |
| flow-specific adaptive attack runner 已实现 | 15 |
| statistical confidence interval reporter 已实现 | 15 |
| full_paper result checker 已实现 | 15 |
| reviewer evidence index builder 已实现 | 10 |
| artifact rebuild 与 claim audit 全链路通过 | 5 |

解释:

```text
90-100: 可进入 full_paper 前最终 dry-run
75-89: 可进入 validation-scale, 但仍需补齐部分 gate
60-74: 只能作为实验协议验证阶段
<60: 不应进入论文主结果生产
```

若某个项目只有文档描述, 没有 repository checker、runner 或 reporter, 则该项目记 0 分。该评分用于防止把“手册完整”误判为“实验系统已经具备 full_paper 产出能力”。
