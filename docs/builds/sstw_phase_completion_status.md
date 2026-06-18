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
| `motion_threshold_calibration` | `structure_planned / protocol_required / external_validation_required` | 当前 formal motion gate 已显式记录 heuristic threshold 与阻塞原因 | 使用独立 calibration split 统计测算并冻结 motion threshold。 |
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

## 4. 2026-06-18 阶段状态更新: motion threshold calibration 后置化

### 4.1 当前推进决策

当前项目允许继续推进 small-scale claim pilot 的其他步骤, 但 formal motion gate 的阈值边界必须显式降级为 pilot guardrail。当前阈值:

```text
motion_delta_threshold: 0.0005
threshold_id: motion_delta_heuristic_v1
threshold_source_split: heuristic_precalibration
usage: pilot_guardrail
```

该阈值不是通过大量样本统计校准得到的正式阈值, 不得用于支撑最终论文级 motion claim。

### 4.2 当前 pilot 可继续推进的内容

在保留上述限制的情况下, 可以继续推进:

```text
attack matrix
negative family
wrong_sampler_replay
wrong_key / without-key control
path_marginal_gain_at_fixed_fpr
replay_uncertainty_mean
claim_support_status governance
```

### 4.3 后续必须补齐的 calibration gate

`motion_threshold_calibration` 已被登记为后续必须进行步骤。正式 claim 或 submission package 前必须完成:

```text
独立 calibration split
negative_static tail 统计
frozen threshold artifact
threshold_source_split 记录
test_time_threshold_update_blocked = true
heldout evaluation split 复核
```

在该 gate 完成前, 阶段状态应保持:

```text
small_scale_claim_pilot_gate: 可继续推进, 但 formal motion claim 不完全闭合
generative_video_model_probe: 可工程探索, 但不得冻结最终 motion-threshold claim
```

### 4.4 small-scale claim pilot gate checker 状态

已新增 small-scale claim pilot gate 自动审计器。该 checker 对当前 Google Drive pilot run 的结论是:

```text
pilot_gate_decision: FAIL
claim_support_status: workflow_progression_only
prompt_count: 8
seed_per_prompt_min: 2
missing_pilot_requirements: 8
```

已满足的部分是 Wan2.1 pilot 生成覆盖和 proxy workflow progression。尚未满足的部分是 attack matrix、negative family、method variant、path marginal gain、negative tail、wrong-key separation、wrong-sampler replay 和 replay uncertainty。因此下一步应实现 pilot postprocess / runner, 不应进入 full experiment。

### 4.5 small-scale claim pilot matrix proxy 补齐状态

已新增并运行 pilot matrix proxy postprocess。当前 Google Drive pilot run 的矩阵缺口已由 governed proxy records 补齐:

```text
pilot_matrix_record_count: 480
attack_count: 3
negative_family_count: 4
method_variant_count: 10
path_marginal_gain_at_fixed_fpr: 0.075
negative_tail_status: not_inflated
wrong_key_score_separation_passed: true
wrong_sampler_replay_control_not_equivalent: true
replay_uncertainty_mean: 0.073608
missing_pilot_requirements: []
```

当前 gate 仍保持:

```text
pilot_gate_decision: FAIL
claim_support_status: blocked_until_motion_threshold_calibration
```

该状态是预期的治理结果: 当前矩阵证据级别为 `proxy_postprocess`, 可以支撑 workflow progression, 但不得替代真实攻击运行和 calibrated motion threshold。

### 4.6 runtime video-file attack runner 状态

已对当前 Google Drive Wan2.1 pilot run 执行 runtime video-file attack runner。当前状态:

```text
runtime_attack_decision: PASS
runtime_attack_record_count: 48
runtime_attack_ready_count: 48
runtime_attack_count: 3
small_scale_pilot_missing_requirement_count: 0
small_scale_pilot_claim_support_status: blocked_until_motion_threshold_calibration
```

该结果说明真实文件级 attack runner 与 attacked video 落盘链路已经具备工程可运行性。当前仍不应进入 full experiment, 因为 formal motion threshold 仍为 heuristic guardrail, 且 runtime attacked videos 尚未接入正式 detection / scoring 表。

## 5. 2026-06-18 阶段状态更新: runtime attack 到 runtime detection 工程闭环

### 5.1 当前工程闭环判定

截至 `2026-06-18T15:34:39Z` 批次, 当前 Wan2.1 small-scale pilot 的工程链路已经从真实生成视频推进到 runtime attack 和 runtime detection 的完整落盘闭环:

```text
generation_records -> videos -> runtime_attack_records -> attacked_videos -> runtime_detection_records -> pilot_gate -> package_manifest
```

最新 Google Drive package 为:

```text
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260618_153437_e90e82ae.zip
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260618_153437_e90e82ae_package_manifest.json
```

### 5.2 最新工程证据

```text
runtime_attack_decision: PASS
runtime_attack_record_count: 48
runtime_attack_ready_count: 48
runtime_detection_decision: PASS
runtime_detection_record_count: 48
runtime_detection_ready_count: 48
runtime_detection_detectable_count: 48
runtime_detection_score_mean: 0.781174
attacked_videos_count_in_package: 48
small_scale_pilot_missing_requirement_count: 0
```

### 5.3 仍然不能声明的内容

该闭环只说明工程层面的真实文件级 attack 和 detection 路径已经接通。当前仍不能声明 final paper claim 已成立, 原因是:

```text
pilot_gate_decision: FAIL
claim_support_status: blocked_until_motion_threshold_calibration
motion_threshold_source_split: heuristic_precalibration
motion_threshold_calibration_required: true
```

因此当前项目状态应表述为:

```text
工程层面: generation / attack / detection / package 已闭合。
claim 层面: 仍阻塞于 motion_threshold_calibration, 不得进入 final claim 或 full experiment 结论冻结。
```

## 6. 2026-06-19 阶段状态更新: motion threshold calibration 已执行但未通过

### 6.1 当前 calibration artifact 状态

已新增并运行 `motion_threshold_calibration` 工程入口。最新 Google Drive package 为:

```text
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260618_162447_882754a4.zip
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260618_162447_882754a4_package_manifest.json
```

package manifest 已包含 calibration summary:

```text
motion_threshold_calibration_decision: INSUFFICIENT_SAMPLE
motion_threshold_id: motion_delta_heuristic_v1
motion_threshold_source_split: heuristic_precalibration
motion_threshold_calibration_required: true
```

### 6.2 当前总体状态

```text
工程层面: motion threshold calibration runner / records / threshold artifact / report / package summary 已闭合。
统计层面: 当前 pilot run 样本不足, 不能冻结 calibrated threshold。
claim 层面: small_scale_claim_pilot_gate 仍为 FAIL, 阻塞原因仍是 motion_threshold_calibration。
```

该状态是预期的治理结果, 因为当前 16 条 pilot main records 不能替代独立 calibration split。
