# synthetic_state_inference_sanity 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段用于在可控 synthetic latent 中验证密钥条件状态空间推断是否优于普通时序聚合器、显式时间对齐和通用状态模型。该阶段是方法 sanity, 不是最终真实视频生成主 claim。

### 1.2 输入

```text
configs/protocol/synthetic_state_inference.json
configs/methods/method_variants_synthetic_state.json
configs/attacks/synthetic_temporal_attacks.json
main/methods/state_space_watermark/
main/baselines/
experiments/synthetic_state_inference/
```

### 1.3 构建任务

1. 构造 synthetic latent 与可控 key-conditioned state trace。
2. 生成 calibration negative、test negative、clean positive、attacked positive 和 wrong-key control。
3. 用 calibration negative 固定 `target_fpr = 0.01` 的阈值。
4. 比较 SSTW state posterior 与普通 temporal aggregator、explicit temporal alignment、generic SSM、key-agnostic SSM。
5. 在 temporal crop、frame dropping、duplication、speed change、local clip 和 noise corruption 下评估。

### 1.4 必须 baseline

```text
mean_temporal_aggregator
max_temporal_aggregator
conv1d_temporal_aggregator
explicit_temporal_alignment_baseline
generic_ssm_baseline
key_agnostic_ssm_baseline
wrong_key_control
state_shuffle_control
```

### 1.5 必须记录字段

```text
S_payload_raw
S_state_posterior
S_final
key_id
method_variant
sample_role
split
threshold_value
threshold_source_split
test_time_threshold_update_blocked
key_state_admissibility_status
```

### 1.6 通过标准

1. key-conditioned posterior 在 `TPR@FPR=0.01` 下优于普通 aggregator。
2. wrong-key control 不接近 correct-key positive。
3. 状态搜索不能显著抬高 negative score tail。
4. 结果可由 records 重建为 tables 和 reports。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已经包含 synthetic state inference 相关 runner、mechanism audit、table builder 和 package outputs 模块:

```text
experiments/synthetic_state_inference/build_records.py
experiments/synthetic_state_inference/runner.py
experiments/synthetic_state_inference/mechanism_audit.py
experiments/synthetic_state_inference/table_builder.py
experiments/synthetic_state_inference/package_outputs.py
```

### 2.2 已有方法模块

状态空间水印核心模块已经存在于:

```text
main/methods/state_space_watermark/
```

包括 key conditioner、state transition、state observation、state smoother、state synchronizer 和 admissibility 相关实现。

### 2.3 当前阶段使用边界

该阶段可以支撑“密钥条件状态空间推断在可控 synthetic latent 中成立”的机制证据, 但不能单独支撑“真实 Flow Matching 视频生成轨迹水印”的最终主张。
