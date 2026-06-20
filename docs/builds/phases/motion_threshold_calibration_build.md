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

## 4. 当前通过条件

`motion_threshold_calibration` 只有同时满足以下条件才允许 `PASS`:

```text
negative_static_calibration_count >= 128
positive_motion_calibration_count >= 64
ambiguous_low_motion_calibration_count >= 32
estimated_static_fpr <= target_static_fpr
positive_motion_pass_rate_at_threshold >= 0.80
positive_negative_motion_delta_margin > 0
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

用于检查 positive_motion 最小分数是否高于 clean negative_static 最大分数。若该值小于或等于0, 说明两类样本在当前 motion metric 下发生重叠。

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
5. 当 positive_motion 与 clean negative_static 分数重叠时, 不再允许 calibration PASS。
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
