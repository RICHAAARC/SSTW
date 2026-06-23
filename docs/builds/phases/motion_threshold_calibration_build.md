# motion_threshold_calibration 分阶段构建流程

本文档记录 `motion_threshold_calibration` 阶段的目标、工程入口、通过条件和当前修复状态。该阶段的职责是把早期 heuristic motion gate 升级为可审计、可冻结、可复现的 calibrated motion threshold。

## 1. 阶段目标

该阶段用于回答一个具体问题:

```text
当前 motion_delta_score 是否能够把静止负样本与真实运动正样本稳定区分开。
```

因此它不能只检查样本数量, 还必须检查分数可分性。若 `positive_motion` 在冻结阈值下大面积无法通过, 则说明当前 prompt 设计、生成结果或 motion metric 不足以支撑后续 motion-related claim。

## 2. 样本分组

当前工程约定使用 3 类 calibration split:

```text
negative_static: 明确静止或近似静止的视频, 用于估计静止负样本尾部。
positive_motion: 明确应具有可见运动的视频, 用于验证阈值不会误伤真实运动样本。
ambiguous_low_motion: 低运动或边界运动视频, 用于审计边界行为, 不直接抬高 positive claim。
```

推荐的最小工程规模为:

```text
negative_static: 128+
positive_motion: 64+
ambiguous_low_motion: 32+
```

当前 prompt suite 的构造方式为:

```text
16 negative_static prompts x 8 calibration seeds = 128
8 positive_motion prompts x 8 calibration seeds = 64
4 ambiguous_low_motion prompts x 8 calibration seeds = 32
```

## 3. 阈值冻结原则

必须遵守以下流程:

```text
calibration split -> threshold artifact -> frozen threshold -> evaluation split
```

禁止在 evaluation split 或每次生成后动态重算阈值。若后续需要重新校准, 必须生成新的 `threshold_id`, 不能覆盖旧阈值。

full_paper 阶段还必须区分 engineering motion threshold 与 paper-level detection threshold。`motion_delta_calibrated_v1` 只能证明运动可观测性门控可用于 pilot guardrail, 不能替代 `TPR@FPR=0.001` 的 watermark detection threshold。

full_paper 前必须补齐:

```text
paper_fixed_fpr_calibration_ready
heldout_negative_motion_tail_report
threshold_stability_confidence_interval
prompt_dominance_audit
cluster_by_prompt_threshold_sensitivity
```

## 4. 当前通过条件

`motion_threshold_calibration` 只有同时满足以下条件才允许 `PASS`:

```text
negative_static_calibration_count >= 128
positive_motion_calibration_count >= 64
ambiguous_low_motion_calibration_count >= 32
estimated_static_fpr <= target_static_fpr
positive_motion_pass_rate_at_threshold >= 0.80
motion_threshold_id == motion_delta_calibrated_v1
```

其中:

```text
positive_motion_pass_rate_at_threshold
```

用于防止“阈值虽然控制住静止负样本, 但同时误伤大多数真实运动样本”的情况。

```text
positive_negative_motion_delta_margin
```

用于诊断 positive_motion 最小分数是否高于 clean negative_static 最大分数。由于真实生成模型可能出现少量 stochastic outlier, 该字段不再单独作为硬阻塞条件; 硬阻塞以 `positive_motion_pass_rate_at_threshold` 为准。

## 5. 决策状态

当前 runner 输出 3 类稳定决策:

```text
PASS: 样本数量充足, 静止尾部受控, positive_motion 可分。
INSUFFICIENT_SAMPLE: calibration split 样本数量不足。
FAIL_NOT_SEPARABLE: 样本数量充足, 但 positive_motion 与 negative_static 不可分。
```

当结果为 `FAIL_NOT_SEPARABLE` 时, 后续 small-scale claim pilot 必须继续保持:

```text
claim_support_status: blocked_until_motion_threshold_calibration
motion_threshold_calibration_required: true
```

## 6. 2026-06-20 修复记录

本次修复完成以下变更:

```text
1. 新增 minimum_positive_motion_pass_rate_at_threshold, 默认值为 0.80。
2. 新增 positive_negative_motion_delta_margin。
3. 新增 FAIL_NOT_SEPARABLE 决策。
4. 当 positive_motion_pass_rate_at_threshold 低于门槛时, 不再允许 calibration PASS。
5. positive_negative_motion_delta_margin 保留为诊断字段, 用于识别 outlier 和重叠风险。
6. 增强 calibration prompt suite, 让 positive_motion 明确要求大幅可见位移, 让 negative_static 明确要求 frozen frame。
7. 更新测试用例, 覆盖样本数量充足但分数不可分的失败路径。
```

## 7. 对当前项目推进的影响

最近一次结果中曾出现:

```text
positive_motion_pass_rate_at_threshold = 0.1875
```

该结果表示大多数 `positive_motion` 样本在当前冻结阈值下没有通过。修复后, 这类结果会被明确判定为:

```text
motion_threshold_calibration_decision: FAIL_NOT_SEPARABLE
claim_support_status: blocked_until_motion_threshold_calibration
```

因此下一步不应直接进入 small-scale claim pilot, 而应先重跑 `motion_calibration` profile, 观察增强 prompt 后的 positive_motion 是否能够与 negative_static 分离。

## 8. 需要重跑的最小范围

修复后不需要重跑全部历史流程。最小重跑范围为:

```text
prepare prompt suite
Wan2.1 generation with PROFILE = motion_calibration
formal_metric_runner
motion_threshold_calibration
package_outputs
```

只有当 `motion_threshold_calibration_decision = PASS` 后, 才建议继续推进 small-scale claim pilot。


## 9. 2026-06-21 修复记录: motion metric 与 prompt suite 增强

本次修复完成以下变更:

```text
1. 新增 motion_delta_focus_score, 用高差分区域均值减去中位差分, 降低全局闪烁和曝光漂移对 calibration 的影响。
2. motion_threshold_calibration 优先使用 motion_delta_focus_score, 缺失时回退到历史 motion_delta_score。
3. positive_negative_motion_delta_margin 保留为诊断字段, 不再在 positive pass rate 已达标时单独阻塞。
4. 替换污染较强的 negative_static prompt, 使用更简单的高对比静态几何或静态物体场景。
5. 强化 positive_motion prompt, 使用更大物体、更大位移、更高对比的运动描述。
```

该变更属于工程修复, 需要重新运行:

```text
prepare prompt suite
Wan2.1 generation with PROFILE = motion_calibration
formal_metric_runner
motion_threshold_calibration
package_outputs
```


## 2026-06-22 工程推进: prompt-aware robust calibration 防泄漏协议已落地

本次工程推进将 motion threshold calibration 升级为 prompt-aware robust engineering calibration。核心约束如下:

```text
污染过滤不能依赖 S_final、S_final_conservative、watermark_detection_score 或任何最终判定分数。
污染过滤只能使用 motion observability / prompt validity / visual quality 相关字段。
```

已落地的工程规则:

```text
motion_calibration_score_role: engineering_prompt_audit
contamination_decision_source: motion_observability_score_only
final_detection_score_filtering_blocked: true
no_final_detection_score_used_for_filtering: true
threshold_selection_strategy: prompt_aware_robust_quantile_p95
target_static_fpr_engineering: 0.05
paper_fixed_fpr_calibration_ready: false
not_final_paper_fpr_0_01: true
```

新增 audit artifacts:

```text
records/prompt_contamination_audit_records.jsonl
tables/prompt_contamination_audit_table.csv
artifacts/prompt_contamination_audit.json
artifacts/threshold_stability_audit.json
```

当前阶段 PASS 只表示 engineering motion threshold 可用于后续 small-scale pilot gate, 不表示论文级 `TPR@FPR=0.01` 已完成。论文级 PASS 仍需要更大 held-out negative split, 并在 frozen threshold 下报告 fixed-FPR 结果与置信区间。


## 2026-06-22 修复记录: positive motion 可观测性增强

最新真实 Wan2.1 motion calibration 结果显示, prompt-aware robust calibration 已经能够隔离污染负样本, 但正运动样本仍未达到工程通过门槛:

```text
package_batch_id: 20260622_082859_825b4762
motion_threshold_calibration_decision: FAIL_NOT_SEPARABLE
positive_motion_pass_rate_at_threshold: 0.75
positive_motion_pass_rate_wilson_lower: 0.631835
minimum_positive_motion_pass_rate_at_threshold: 0.80
minimum_positive_motion_pass_rate_wilson_lower: 0.70
```

按 prompt 聚合后, 主要弱项集中在:

```text
motion_calib_positive_motion_00: 3 / 8
motion_calib_positive_motion_02: 1 / 8
motion_calib_positive_motion_04: 6 / 8
motion_calib_positive_motion_06: 6 / 8
```

本次修复不放宽阈值, 也不改变污染过滤协议。修复方向是提升 calibration prompt 的真实运动可观测性:

```text
1. 将抽象几何运动 prompt 替换为更容易被 Wan2.1 执行的真实前景大物体运动。
2. 对 positive_motion_00 / 02 / 04 / 06 增加 close-up、foreground、entire frame、consecutive frames 等可观测约束。
3. 移除高污染或高频纹理的 negative_static prompt, 例如 checkerboard、chess board、wall clock illustration。
4. 将 prompt_suite_id 更新为 generative_video_probe_prompt_suite_motion_observability_repair, 防止与旧 calibration 批次混淆。
```

该修复属于 prompt suite 工程修复。它不能直接证明水印机制有效, 只能为下一次 `PROFILE = motion_calibration` 重跑提供更干净、更高可观测性的 calibration 输入。修复后仍必须重新运行:

```text
prepare prompt suite
Wan2.1 generation with PROFILE = motion_calibration
formal_metric_runner
motion_threshold_calibration
package_outputs
```


## 10. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 已完成 engineering calibration |
| 主要差距项 | 工程阈值可用于 pilot guardrail, 但不是论文级 FPR=0.001 阈值。 |
| 下一步构建方向 | full_paper 前扩展 held-out negative 并报告置信区间和 threshold stability。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 10.1 快速检查清单

```text
stage_status: 已完成 engineering calibration
gap_item: 工程阈值可用于 pilot guardrail, 但不是论文级 FPR=0.001 阈值。
next_action: full_paper 前扩展 held-out negative 并报告置信区间和 threshold stability。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```

