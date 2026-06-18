# motion_threshold_calibration 分阶段构建流程

本文档记录 `motion_threshold_calibration` 阶段的构建流程与当前完成情况。该阶段用于把当前 pilot 中的 heuristic motion gate 升级为可审计、可冻结、可复现的 calibrated motion threshold。本文档不直接支撑论文最终 claim。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段的目标是为 formal motion gate 建立统计校准流程。当前 `motion_delta_threshold = 0.0005` 仅作为 pilot 阶段的 heuristic guardrail, 用于阻止明显低运动视频支撑 motion-related claim。它尚未通过大量 positive / negative / ambiguous 样本统计测算, 因此不能作为最终论文级 motion threshold。

### 1.2 输入样本分组

建议至少区分以下分组:

```text
positive_dynamic: 明确应具有可见运动的视频
negative_static: 明确静态或近似静态的视频
ambiguous_low_motion: 慢速旋转、微弱镜头运动、低纹理运动等边界视频
failure_cases: pilot 中被 heuristic gate 阻塞的样本
```

### 1.3 建议样本规模

small-scale calibration 建议优先采用:

```text
positive_dynamic: 64
negative_static: 128
ambiguous_low_motion: 32
total: 224 videos
```

该规模用于判断当前 heuristic threshold 是否明显过严或过松, 不能替代 full experiment 的论文级 threshold calibration。若要支撑最终 `TPR@FPR=0.01`, negative split 应扩大到 `1000+`, 更稳妥为 `2000-3000`。

### 1.4 阈值冻结原则

正式实验必须遵循:

```text
calibration split -> threshold artifact -> frozen threshold -> evaluation split
```

禁止在 evaluation split 或每次生成后动态重算阈值。若需要重新校准, 必须生成新的 `threshold_id`, 不能覆盖旧阈值。

### 1.5 必须记录字段

后续实现时应至少记录:

```text
threshold_id
threshold_source_split
threshold_value
target_fpr
test_time_threshold_update_blocked
motion_delta_threshold
temporal_flicker_threshold
calibration_negative_count
calibration_positive_count
calibration_ambiguous_count
calibration_commit
```

其中 `test_time_threshold_update_blocked` 必须为 `true`, 以防止 evaluation records 反向参与阈值选择。

### 1.6 通过标准

```text
threshold_source_split 指向独立 calibration split
negative_static tail 未膨胀
empirical FPR <= target_fpr
positive_dynamic recall 不发生不可接受下降
ambiguous_low_motion 行为可解释
threshold artifact 可由 records 重建
```

## 2. 当前阶段完成情况

### 2.1 当前阶段判定

`motion_threshold_calibration` 当前判定为:

```text
structure_planned / protocol_required / external_validation_required
```

该阶段尚未运行。当前代码中的 `motion_delta_threshold = 0.0005` 仅作为:

```text
threshold_id: motion_delta_heuristic_v1
threshold_source_split: heuristic_precalibration
usage: pilot_guardrail
```

### 2.2 对当前 pilot 的影响

当前 small-scale pilot 可以继续推进 attack matrix、negative family、wrong-sampler replay、path marginal gain 和 replay uncertainty 等步骤。但所有涉及 formal motion claim 的结论必须保留以下限制:

```text
formal motion gate currently uses heuristic_precalibration threshold.
calibrated motion threshold remains required before final paper claim.
```

### 2.3 与 full experiment 的关系

在进入 submission package 或最终论文 claim 前, 必须完成本阶段或等价的 calibration artifact。未完成本阶段时, full experiment 可以作为工程探索运行, 但不能把 motion threshold 相关结果声明为 calibrated formal claim。

## 3. 2026-06-19 calibration runner 执行结果

### 3.1 已新增工程入口

当前已新增 formal motion threshold calibration runner:

```text
experiments/generative_video_model_probe/motion_threshold_calibration.py
```

该 runner 读取:

```text
records/generation_records.jsonl
records/formal_quality_motion_semantic_records.jsonl
```

并写出:

```text
records/motion_threshold_calibration_records.jsonl
tables/motion_threshold_calibration_table.csv
thresholds/motion_threshold_calibration_threshold.json
artifacts/motion_threshold_calibration_decision.json
reports/motion_threshold_calibration_report.md
```

### 3.2 当前 Google Drive run 的 calibration 结果

已在当前 Wan2.1 pilot run 上执行 calibration。最新 package 为:

```text
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260618_162447_882754a4.zip
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260618_162447_882754a4_package_manifest.json
```

当前结果为:

```text
motion_threshold_calibration_decision: INSUFFICIENT_SAMPLE
motion_threshold_calibration_ready: false
motion_threshold_id: motion_delta_heuristic_v1
motion_delta_threshold: 0.0005
motion_threshold_source_split: heuristic_precalibration
negative_static_calibration_count: 0
positive_motion_calibration_count: 0
usable_motion_calibration_record_count: 16
motion_calibration_record_count: 16
motion_threshold_calibration_required: true
test_time_threshold_update_blocked: true
```

阻塞原因为:

```text
negative_static_calibration_count_below_min
positive_motion_calibration_count_below_min
```

### 3.3 结论

本次执行完成了 calibration 工程入口和 artifact 闭环, 但没有完成统计意义上的 calibrated threshold。原因不是 runner 失败, 而是当前 pilot run 没有独立 `calibration` split, 也没有足够的 `negative_static` calibration tail。

因此当前 claim gate 仍必须保持:

```text
pilot_gate_decision: FAIL
claim_support_status: blocked_until_motion_threshold_calibration
```

下一步如果要真正解除该阻塞, 需要生成独立 calibration split, 至少包含:

```text
negative_static calibration records >= 8  # 工程最小门槛
positive_motion calibration records >= 8  # 工程最小门槛
```

若目标是支撑论文级 `TPR@FPR=0.01`, negative static / negative low-motion tail 应扩大到 1000+ 样本级别, 而不是使用当前 16 条 pilot main records。
