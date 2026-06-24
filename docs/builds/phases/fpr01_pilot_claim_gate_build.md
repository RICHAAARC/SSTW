# fpr01_pilot_claim_gate 分阶段构建文档

## 1. 阶段定位

`fpr01_pilot_claim_gate` 位于 `validation_scale` 与 `full_paper` 之间。该阶段用于补充一个中等规模真实 Wan2.1 pilot, 目标是让项目能够在 pilot 级别形成合规的 `TPR@FPR=0.01` 结论。

该阶段不等价于 full-paper。它不能支持 `TPR@FPR=0.001`, 也不能生成论文主表或最终 submission package。

## 2. 数据集构造要求

`TPR@FPR=0.01` pilot 必须采用与 full-paper 同构的固定阈值协议:

```text
calibration split
-> frozen threshold artifact
-> held-out test split
-> tables / figures / claim audit
```

当前仓库已在 prompt suite 中新增独立 `fpr01_pilot` profile。该 profile 不复用 `pilot` 或 `validation_scale` prompt, 避免样本角色混淆。

```text
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

## 3. 工程入口

```text
configs/protocol/fpr01_pilot_generative_probe.json
experiments/generative_video_model_probe/fpr01_pilot_gate.py
experiments/generative_video_model_probe/colab_runtime.py PROFILE = fpr01_pilot
paper_workflow/notebook_utils/generative_video_model_probe_workflow.py::build_fpr01_pilot_gate_command
paper_workflow/colab_utils/generative_video_model_probe_colab.ipynb
scripts/package_results/generative_video_drive_packager.py
```

## 4. Gate 输出

该 gate 写出以下 governed artifacts:

```text
records/fpr01_pilot_gate_records.jsonl
tables/fpr01_pilot_gate_table.csv
thresholds/fpr01_pilot_frozen_threshold.json
artifacts/fpr01_pilot_gate_decision.json
reports/fpr01_pilot_gate_report.md
```

package manifest 会同步记录:

```text
fpr01_pilot_gate_decision
fpr01_pilot_claim_support_status
fpr01_threshold_protocol
fpr01_threshold_source_split
fpr01_test_time_threshold_update_blocked
fpr01_tpr_at_fpr_01
fpr01_calibration_negative_fpr_at_threshold
fpr01_heldout_negative_fpr_at_threshold
fpr01_calibration_negative_event_count
fpr01_heldout_test_negative_event_count
fpr01_heldout_attacked_positive_event_count
fpr01_tpr_at_fpr_01_pilot_claim_allowed
fpr01_tpr_at_fpr_001_claim_allowed
```

## 5. 通过标准

```text
fpr01_prompt_count >= 21
fpr01_seed_per_prompt_min >= 8
fpr01_calibration_seed_per_prompt_min >= 4
fpr01_test_seed_per_prompt_min >= 4
fpr01_unique_video_count >= 168
fpr01_calibration_unique_video_count >= 84
fpr01_test_unique_video_count >= 84
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
small_scale_claim_pilot_gate_decision == PASS
```

## 6. Claim 边界

通过该 gate 只允许写为:

```text
claim_support_status: fpr01_pilot_calibrated_heldout_claim_ready
tpr_at_fpr_01_pilot_claim_allowed: true
tpr_at_fpr_001_claim_allowed: false
full_paper_allowed: false
```

这表示 pilot 级 `TPR@FPR=0.01` 结论已经使用 calibration / held-out split 与冻结阈值协议产生。它的合规性来自实验协议, 与 full-paper 的差异主要是样本规模。

禁止将该阶段解释为:

```text
TPR@FPR=0.001 已成立
full-paper fixed-FPR 结果已成立
现代 external baseline 主表对比已完成
内部消融主表已完成
submission package 已冻结
```

## 7. 当前完成状态

当前仓库层面已经完成 `fpr01_pilot` 的数据集构造、profile 接入、frozen threshold gate runner、notebook workflow 接入、package manifest 摘要和默认 pytest 覆盖。真实 Wan2.1 GPU 结果尚未生成, 因此该阶段的实验结论仍需用户在 Colab 中以 `PROFILE = 'fpr01_pilot'` 复跑后再审计。
