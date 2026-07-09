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

### 1.1 五层结果产出梯度

顶刊顶会实验不应把“方法能运行”“小样本能跑通”“FPR=10% 小样本论文闭合”“pilot 级论文结果”和“正式 full_paper 结果”混为一类。本项目统一采用以下五层结果产出梯度:

| 层级 | 目标 FPR | 样本规模 | 主要作用 | 是否允许支撑效果主张 |
|---|---:|---|---|---|
| `method_mechanism_validation` | 不固定 | 最小机制样本 | 验证 SSTW 方法机制、状态空间证据、轨迹观测和基础 runner 是否可运行。 | 否 |
| `validation_scale` | `0.10` | 小样本 | 小样本全流程打通验证, 属于 paper 级前的全流程打通层。它必须产出与论文协议同构的 records、tables、figures、reports、manifests、baseline、消融、attack、CI 和 artifact rebuild 文件, 用于提前发现 pilot_paper / full_paper 的阻断。 | 否 |
| `probe_paper` | `0.10` | 小样本论文闭合协议 | 使用 pilot_paper 级样本结构和同构 attack / baseline / 消融 / 图表协议, 在 FPR=10% 下验证论文结论是否能够闭合。 | 仅允许 FPR=10% 小样本论文闭合主张 |
| `pilot_paper` | `0.01` | 小规模论文协议 | 以较小成本产出 FPR=1% 级别的代表性 paper 协议结果, 检查真实模型、baseline、消融和 fixed-FPR 统计是否具备扩展到 full_paper 的可报告性。 | 仅允许 pilot 级主张 |
| `full_paper` | `0.001` | 正式规模论文协议 | 产出 FPR=0.1% 级别正式论文主结果, 包含主表、主图、CI、claim audit、artifact rebuild 和 reviewer evidence index。 | 是 |

`validation_scale` 是进入 `probe_paper`、`pilot_paper` 和后续 `full_paper` 流程的硬门禁。它通过只说明论文协议产物链路已经小样本跑通, 不说明 SSTW 的效果已经达到投稿主张。进入 `probe_paper` 必须在 `validation_scale` PASS 后生成 `validation_scale_to_probe_paper_transition_decision`; 进入 `pilot_paper` 还必须在 `probe_paper` PASS 后生成 `probe_paper_to_pilot_paper_transition_decision`。`validation_scale` 与 `probe_paper` 都是 `full_paper` 的必要条件但不是充分条件; `full_paper` 仍需要 `pilot_paper_gate`、`pilot_paper_to_full_paper_transition_decision`、`full_paper_result_checker`、CI、claim audit、artifact rebuild 和 submission freeze 相关检查通过。

主干门禁只保留:

```text
protocol_governance -> mechanism_validation -> validation_scale -> probe_paper -> pilot_paper -> full_paper -> submission_freeze
```

历史 `small_scale_claim_pilot_gate` 只作为 `mechanism_validation` 下的小样本机制检查记录, 不再作为主干门禁。`generative_video_model_probe` 只表示真实生成式视频模型实现 package, 不再作为独立门禁。

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

这些要求必须在 `configs/protocol/full_paper_generative_probe.json` 中以协议配置形式登记。Notebook、临时 CLI 参数或手工记录不能降低该配置中的样本量、阈值来源和 CI 要求。

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
metric_status
```

进入正式主表的 modern external baseline 必须统一使用:

```text
metric_status == measured_formal
```

外部 baseline 的正式结果必须由本项目自包含产出:

```text
project_clone
project_build
project_run
project_adapt
project_record
```

允许本项目在 Colab 或受治理运行环境中 clone GitHub 官方代码、安装依赖、下载公开权重、调用官方 API 或官方命令, 但最终 records、tables 和 reports 必须由本项目流程写出。禁止把外部补交的 result bundle、手写 JSON、NPZ 分数文件、论文表格数字或 SSTW proxy 分数作为主表 baseline 结果。governed non-run record 只能解释阻断原因, 不能替代正式 `measured_formal` baseline。

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
compression_attack: H.264 / H.265 / MPEG-4 / platform_transcode_proxy, 多 CRF 或 bitrate 强度
temporal_crop_attack
temporal_resampling_attack
frame_drop_insert_swap_average_speed_or_irregular_attack
spatial_transform_attack: crop / resize / rotation / perspective
visual_degradation_attack: Gaussian noise / salt-and-pepper / blur / median blur / denoise / brightness / contrast / gamma / color jitter / sharpen
combined_attack: compression + crop, compression + brightness, compression + temporal disturbance, compression + noise, compression + color jitter, crop + rotation
generative_recompression_or_regeneration_attack
endpoint_preserving_path_perturbation_attack
flow_time_grid_mismatch_attack
wrong_sampler_replay_attack
wrong_prompt_replay_attack
wrong_key_attack
detector_probing_with_public_negatives
watermark_removal_optimization_attack
watermark_spoofing_or_copy_attack
collusion_multi_sample_attack
adversarial_detector_evasion_attack
```

当前工程协议采用三层 attack 要求:

- `validation_scale`: `video_compression_runtime`、`temporal_crop_runtime`、`frame_rate_resampling_runtime`, 只用于小样本全流程打通和公平比较门禁。
- `pilot_paper`: 使用代表性压缩、时间、空间和视觉退化 attack manifest。其作用是以较小样本提前暴露 official baseline 适配、fixed-FPR 统计和 attack 覆盖缺口。
- `full_paper`: 使用多强度压缩、完整时间扰动、空间几何、视觉退化、组合攻击和非 runtime 自适应攻击协议。此层级才可支撑顶会 / 顶刊级“完整鲁棒性”讨论。

`pilot_paper` 和 `full_paper` 的 runtime attack manifest 都必须从 protocol config 读取, 不应由 Notebook 手工缩减。若需要临时调试, 只能使用 helper 或 dry-run profile, 其产物不得进入 paper gate。

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
| 只有 pilot, 没有 validation_scale 与 full_paper_result_checker | 不允许 full_paper |
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
| pilot_paper gate 已实现并测试通过 | 15 |
| modern external baseline self-contained runner 已实现, 且 non-run record 仅作为阻断记录而不替代 measured_formal | 15 |
| internal ablation matrix 已实现并能重建表格 | 10 |
| flow-specific adaptive attack runner 已实现 | 15 |
| statistical confidence interval reporter 已实现 | 15 |
| full_paper_result_checker 已实现 | 15 |
| reviewer evidence index builder 已实现 | 10 |
| artifact rebuild 与 claim audit 全链路通过 | 5 |
| stage transition / external baseline self-containment / data split leakage 三个轻量判定已实现 | 5 |

解释:

```text
90-100: 可作为进入下一主干阶段前的工程化预检, 仍必须遵守 protocol_governance -> mechanism_validation -> validation_scale -> probe_paper -> pilot_paper -> full_paper -> submission_freeze 顺序
75-89: 可进入 validation_scale, 但仍需补齐部分 gate
60-74: 只能作为实验协议验证阶段
<60: 不应进入论文主结果生产
```

若某个项目只有文档描述, 没有 repository checker、runner、reporter 或轻量 decision artifact, 则该项目记 0 分。评分表原始项合计为 105 分, 最后一项为加分型治理项; 对外报告 readiness 时必须写出 `raw_readiness_score` 和 `normalized_readiness_score = min(raw_readiness_score, 100)`, 不得用加分项绕过任何主干门禁。该评分用于防止把“手册完整”误判为“实验系统已经具备 full_paper 产出能力”。
