# SSTW 分阶段完成情况记录

## 0. 文档定位

本文档用于记录 SSTW 各阶段在当前仓库中的工程状态、证据状态和后续缺口。它不是论文 claim 的直接来源。论文中的 supported claims 必须由 governed records、tables、figures、reports 或 manifests 支撑。

## 1. 阶段状态分级

| 状态值 | 含义 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| `structure_ready` | 阶段所需目录、配置、runner、checker、packager、文档或基础模块已经建立。 | 该阶段具备继续实现和运行的工程入口。 | 不能说明核心机制已经有效, 也不能说明实验结果已经成立。 |
| `mechanism_ready` | 阶段核心机制已有可运行或可测试实现。 | 该阶段具备机制级测试或消融的代码基础。 | 不能说明真实模型、真实数据或 fixed-FPR 结果已经达到论文目标。 |
| `protocol_ready` | 阶段已有固定 split、sample role、threshold、baseline、control、checker 或 manifest 约束。 | 该阶段的实验协议有明确边界。 | 不能说明协议下已经产出充分结果。 |
| `artifact_ready` | 阶段已有从 records 重建 tables、reports、packages 或 manifests 的入口。 | 该阶段具备 artifact rebuild 通道。 | 不能说明 artifacts 中已经有最终论文结果。 |
| `external_validation_required` | 阶段仍需要 Colab、GPU、真实视频模型、真实 VAE 链路或外部 baseline 运行结果继续验证。 | 本地 CPU 或 synthetic 结果无法完全覆盖该阶段证据需求。 | 不能说明阶段失败, 只说明还需要外部运行环境补齐证据。 |

## 2. 阶段完成情况总览

| 阶段 | 当前状态 | 主要依据 | 后续重点 |
|---|---|---|---|
| `protocol_governance_foundation` | `structure_ready / protocol_ready` | configs、field registry、harness、constraint tests | 随新增字段、negative family 和旧字段映射继续同步注册。 |
| `flow_model_adapter_preflight` | `structure_ready / mechanism_ready / protocol_ready / artifact_ready` | Wan2.1 Colab preflight、sampler signature、time grid、latent displacement proxy、Drive package | 作为后续 sampling-time 和 pilot 的真实模型接口前置证据。 |
| `sampling_time_constraint_probe` | `structure_ready / mechanism_ready / protocol_ready / artifact_ready` | Wan2.1 recommended profile、constraint records、postprocess checker、Drive package | 已完成机制前置验证, 下一步进入 small-scale claim pilot。 |
| `small_scale_claim_pilot_gate` | `structure_ready / protocol_ready / external_validation_required` | sampling-time recommended 已通过, generative probe 与 checker 入口已存在 | 运行 pilot split, 验证 attack、negative family、wrong-sampler replay 与 fixed-FPR path marginal gain。 |
| `generative_video_model_probe` | `structure_ready / protocol_ready / external_validation_required` | generative probe runner、external baseline runner、Colab 入口 | 仅在 pilot gate 通过后进入 full experiment。 |
| `replay_and_authenticated_sketch_gate` | `structure_ready` | digest、manifest、trajectory trace 基础模块 | 补齐 authenticated sketch、replay uncertainty、wrong prompt replay 与 checker。 |
| `submission_package_freeze` | `structure_ready / artifact_ready` | submission freeze runner、main tables、readiness summary | 等待 pilot / full experiment governed records 后再冻结最终 artifacts。 |

## 3. 2026-06-18 阶段状态更新: 机制前置验证闭合

### 3.1 当前总体判定

截至 `2026-06-18T02:34:47Z` 批次, 项目已经完成真实 Wan2.1 路径上的机制前置验证闭合。这里的“机制前置验证闭合”指以下内容已经具备 governed records、checker 与 package 证据:

```text
protocol_governance_foundation: structure_ready / protocol_ready
flow_model_adapter_preflight: structure_ready / mechanism_ready / protocol_ready / artifact_ready
sampling_time_constraint_probe: structure_ready / mechanism_ready / protocol_ready / artifact_ready
small_scale_claim_pilot_gate: structure_ready / protocol_ready / external_validation_required
```

该判定不能等同于“所有论文 claim 已经完成”。当前尚未完成 small-scale claim pilot, 因此不能进入 full experiment, 也不能声明最终 `TPR@FPR=0.01`、baseline comparison、ablation table 或 submission claim 已经成立。

### 3.2 已完成的机制证据

最新通过的 sampling-time constraint recommended package 为:

```text
G:\我的云端硬盘\SSTW\packages\sampling_time_constraint\sampling_time_constraint_colab_20260618_023447_f325e2a5.zip
G:\我的云端硬盘\SSTW\packages\sampling_time_constraint\sampling_time_constraint_colab_20260618_023447_f325e2a5_package_manifest.json
```

该批次对应:

```text
package_batch_id: 20260618_023447_f325e2a5
profile: recommended
generation_record_count: 20
constraint_record_count: 320
prompt_count: 2
seed_count: 2
method_variant_count: 5
primary_model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
gpu: NVIDIA L4
```

当前 checker 判定为:

```text
implementation_evidence_status: PASS
mechanism_evidence_status: PASS
missing_mechanism_requirements: []
```

关键机制指标为:

```text
keyed_constraint_alignment_gain_mean: 0.001680
keyed_flow_velocity_alignment_gain_mean: 0.020683
key_separation_gain_over_control: 0.001680
key_separation_flow_velocity_gain_over_control: 0.020685
minimum_key_separation_gain: 0.0005
minimum_key_separation_flow_velocity_gain: 0.0005
```

方向审计结果显示 wrong-key 与 without-key control 不再与 matched-key evidence direction 高相关:

```text
keyed application_evidence_direction_cosine: 1.0
without-key application_evidence_direction_cosine mean: -0.000548
wrong-key application_evidence_direction_cosine mean: -0.000084
```

### 3.3 当前仍未完成的阶段

以下阶段仍保持未完成或仅工程入口就绪状态:

```text
small_scale_claim_pilot_gate: 尚未运行 pilot split, 不能证明 attack / negative family / wrong-sampler replay 下的 claim 稳定性
generative_video_model_probe: 尚未进入 full experiment, 不能产生最终主论文表格
replay_and_authenticated_sketch_gate: authenticated sketch 与 replay uncertainty 仍需进一步闭合
submission_package_freeze: 只能等待 pilot / full experiment governed records 后再冻结
```

### 3.4 阶段推进规则

下一步允许进入:

```text
small_scale_claim_pilot_gate
```

下一步不允许直接进入:

```text
generative_video_model_probe full experiment
submission_package_freeze final claim
```

除非 small-scale claim pilot 证明:

```text
path_marginal_gain_at_fixed_fpr > 0
negative tail 没有膨胀
wrong_sampler_replay 不能伪造正确轨迹
quality_guard 通过
wrong_key / without-key control 保持分离
```

