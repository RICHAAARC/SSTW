# pilot_paper FPR=0.01 claim gate 分阶段构建文档

## 1. 阶段定位

`pilot_paper` 是小样本论文级结果层级, 不是仅用于 workflow progression 的工程预演。该阶段位于 `validation_scale` 与 `full_paper` 之间, 目标是在真实 Wan2.1 上产出可审计的 pilot-scale paper claim, 重点报告 `TPR@FPR=0.01`。

`pilot_paper` 与 `full_paper` 的核心区别只允许是样本规模和统计置信度。二者必须共享同一类论文级协议:

```text
calibration split
-> frozen threshold artifact
-> held-out test split
-> governed records
-> tables / figures / claim audit
```

因此, 通过该阶段可以写成 `pilot_paper` 级论文主张; 但不能外推为 `TPR@FPR=0.001` 或 full_paper 规模主张。

## 2. 数据集构造要求

当前仓库只保留 `pilot_paper` 作为 paper 级小样本正式 profile。运行入口为:

```text
PROFILE = pilot_paper
```

`pilot_paper` 使用独立 prompt / seed 数据集。该数据集不复用 `pilot` 或 `validation_scale` prompt, 避免样本角色混淆。

```text
paper_result_level: pilot_paper
paper_protocol_level: paper_grade_protocol
paper_protocol_difference_from_full_paper: sample_scale_target_fpr_and_attack_coverage
prompt_count: 21
seed_per_prompt: 8
calibration_seed_per_prompt: 4
test_seed_per_prompt: 4
target_generation_video_count: 168
target_runtime_attack_count: 3
target_negative_family_count: 4
target_calibration_negative_event_count: 1008
target_heldout_test_negative_event_count: 1008
target_heldout_attacked_positive_event_count: 252
threshold_protocol: calibration_split_to_frozen_threshold_to_heldout_test_split
```

其中 calibration split 只用于估计并冻结 `FPR=0.01` 阈值, held-out test split 只用于报告 negative FPR 和 attacked positive TPR。test split 运行过程中不得更新阈值。

## 2.1 baseline 与内部消融前置要求

`pilot_paper` 本质是小规模跑完整 full paper 协议。因此在执行 `pilot_paper_gate` 之前, 同一批 held-out test trace 必须已经写出以下 governed artifacts:

```text
records/external_baseline_score_records.jsonl
artifacts/external_baseline_comparison_decision.json
records/validation_internal_ablation_records.jsonl
artifacts/validation_internal_ablation_decision.json
```

`pilot_paper` 需要完整 external baseline 集合, 不能只接入一个现代 baseline, 也不能用显式同步 control 替代现代视频水印 baseline。必须覆盖:

```text
explicit_dtw_temporal_alignment
explicit_frame_matching_temporal_registration
videoshield
vidsig
videoseal
```

其中显式 DTW 与 frame matching 只能写出 `measured_proxy` control records; 5 个主实验现代视频水印 baseline 必须通过项目内 clone / build / run / adapt / record 和正式 adapter 写出 `metric_status = measured_formal` records。内部消融矩阵至少需要覆盖 `sstw_full_method`、endpoint-only、trajectory-only、去 velocity constraint、去 endpoint-aware control、去 replay uncertainty weighting、去 admissibility 和 generic SSM baseline。若任何现代 baseline 或消融缺失, `pilot_paper` gate 必须失败, 不允许先报告 `TPR@FPR=0.01` 再补表。


## 3. 工程入口

```text
configs/protocol/pilot_paper_generative_probe.json
experiments/generative_video_model_probe/pilot_paper_gate.py
experiments/generative_video_model_probe/colab_runtime.py PROFILE = pilot_paper
paper_workflow/notebook_utils/generative_video_model_probe_workflow.py::build_pilot_paper_gate_command
paper_workflow/colab_notebooks/generative_video_runtime_colab.ipynb
scripts/package_results/generative_video_drive_packager.py
```

## 4. Gate 输出

该 gate 写出以下 governed artifacts:

```text
records/pilot_paper_gate_records.jsonl
tables/pilot_paper_gate_table.csv
thresholds/pilot_paper_frozen_threshold.json
artifacts/pilot_paper_gate_decision.json
reports/pilot_paper_gate_report.md
```

文件名和 artifact 字段统一使用 `pilot_paper`; artifact 内部必须写出 `paper_result_level = pilot_paper` 和 `paper_protocol_difference_from_full_paper = sample_scale_target_fpr_and_attack_coverage`。

package manifest 会同步记录:

```text
pilot_paper_gate_decision
pilot_paper_claim_support_status
pilot_paper_result_level
pilot_paper_protocol_level
pilot_paper_protocol_difference_from_full_paper
pilot_paper_protocol_matches_full_paper
pilot_paper_claim_allowed
pilot_paper_external_baseline_trace_count
pilot_paper_external_baseline_trace_count_min
pilot_paper_internal_ablation_trace_count_min
pilot_paper_missing_external_baseline_adapter_names
pilot_paper_missing_internal_ablation_variants
pilot_paper_tpr_at_fpr_01
pilot_paper_calibration_negative_fpr_at_threshold
pilot_paper_heldout_negative_fpr_at_threshold
pilot_paper_calibration_negative_event_count
pilot_paper_heldout_test_negative_event_count
pilot_paper_heldout_attacked_positive_event_count
pilot_paper_tpr_at_fpr_001_claim_allowed
```

## 5. 通过标准

```text
pilot_paper_prompt_count >= 21
pilot_paper_seed_per_prompt_min >= 8
pilot_paper_calibration_seed_per_prompt_min >= 4
pilot_paper_test_seed_per_prompt_min >= 4
pilot_paper_unique_video_count >= 168
pilot_paper_calibration_unique_video_count >= 84
pilot_paper_test_unique_video_count >= 84
calibration_negative_event_count >= 1000
heldout_test_negative_event_count >= 1000
heldout_attacked_positive_event_count >= 200
calibration_negative_family_count >= 4
heldout_negative_family_count >= 4
calibration_negative_event_count_per_family_min >= 200
heldout_negative_event_count_per_family_min >= 200
attack_event_count_per_attack_min >= 60
frozen_threshold_artifact_computable == true
heldout_negative_fpr_at_threshold <= 0.01
tpr_at_fpr_01 is computable
path_marginal_gain_at_fixed_fpr > 0
negative_tail_status == not_inflated
wrong_sampler_replay_control_not_equivalent == true
motion_threshold_calibration_ready == true
method_mechanism_validation_decision == PASS
validation_scale_gate_decision == PASS
validation_scale_to_pilot_paper_transition_decision == PASS
data_split_and_leakage_guard_decision == PASS
external_baseline_comparison_decision == PASS
external_baseline_self_containment_decision == PASS
external_baseline_measured_adapter_count >= 7
modern_external_baseline_formal_measured_adapter_count >= 5
required_external_baseline_adapter_names covered
required_modern_external_baseline_adapter_names measured_formal
pilot_paper_external_baseline_trace_count_min >= 84
validation_internal_ablation_decision == PASS
internal_ablation_variant_count >= 8
required_internal_ablation_variants covered
pilot_paper_internal_ablation_trace_count_min >= 84
```

## 6. Claim 边界

通过该 gate 允许写为:

```text
paper_result_level: pilot_paper
paper_protocol_difference_from_full_paper: sample_scale_target_fpr_and_attack_coverage
claim_support_status: pilot_paper_calibrated_heldout_claim_ready
pilot_paper_claim_allowed: true
tpr_at_fpr_01_pilot_claim_allowed: true
tpr_at_fpr_001_claim_allowed: false
full_paper_allowed: false
```

这表示 pilot_paper 级 `TPR@FPR=0.01` 结论已经使用 calibration / held-out split 与冻结阈值协议产生, 且同批 held-out test trace 已覆盖 external_baseline comparison records 和内部消融矩阵。它是论文级小样本主张, 不是 workflow-only 证据。full_paper 对同一协议进行更大样本规模、更低 FPR 和更强统计置信度的扩展验证。

禁止将该阶段解释为:

```text
TPR@FPR=0.001 已成立
full_paper 规模 fixed-FPR 结果已成立
现代 external baseline full-scale 主表对比已完成
full-scale 内部消融主表已完成
submission package 已冻结
```

## 7. 当前完成状态

当前仓库层面已经完成 `pilot_paper` 的数据集构造、profile 接入、frozen threshold gate runner、external_baseline comparison 前置检查、内部消融矩阵前置检查、notebook workflow 接入、package manifest 摘要和默认 pytest 覆盖。真实 Wan2.1 GPU 结果尚未生成, 因此该阶段的实验结论仍需用户在 Colab 中以 `SSTW_WORKFLOW_PROFILE=pilot_paper` 复跑后再审计。
