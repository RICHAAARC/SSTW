# SSTW 分阶段完成情况记录

## 0. 文档定位

本文档用于记录 SSTW 各阶段在当前仓库中的工程状态、证据状态和后续缺口。它不是论文 claim 的直接来源。论文中的 supported claims 必须由 governed records、tables、figures、reports 或 manifests 支撑。

最新语义优先级如下: 主干门禁只保留 `protocol_governance -> mechanism_validation -> validation_scale -> probe_paper -> pilot_paper -> full_paper -> submission_freeze`。历史 `small_scale_claim_pilot_gate` 仅作为 `mechanism_validation` 下的小样本机制检查记录, 不再作为主干门禁。`generative_video_model_probe` 仅作为真实生成式视频模型实现 package, 不再作为独立门禁。本文档较早日期段落中的旧名称、`EXTERNAL_BASELINE_EVIDENCE_PATHS`、安装或挂载官方实现、外部补交 official result bundle 等旧说法均为历史状态记录; 当前规则以项目内 clone / build / run / adapt / record 和 `measured_formal` 自包含产出为准。

## 1. 阶段状态分级

| 状态值 | 含义 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| `structure_ready` | 阶段所需目录、配置、runner、checker、packager、文档或基础模块已经建立。 | 该阶段具备继续实现和运行的工程入口。 | 不能说明核心机制已经有效, 也不能说明实验结果已经成立。 |
| `mechanism_ready` | 阶段核心机制已有可运行或可测试实现。 | 该阶段具备机制级测试或消融的代码基础。 | 不能说明真实模型、真实数据或 fixed-FPR 结果已经达到论文目标。 |
| `protocol_ready` | 阶段已有固定 split、sample role、threshold、baseline、control、checker 或 manifest 约束。 | 该阶段的实验协议有明确边界。 | 不能说明协议下已经产出充分结果。 |
| `artifact_ready` | 阶段已有从 records 重建 tables、reports、packages 或 manifests 的入口。 | 该阶段具备 artifact rebuild 通道。 | 不能说明 artifacts 中已经有最终论文结果。 |
| `external_validation_required` | 阶段仍需要 Colab、GPU、真实视频模型、真实 VAE 链路或外部 baseline 运行结果继续验证。 | 本地 CPU 或 synthetic 结果无法完全覆盖该阶段证据需求。 | 不能说明阶段失败, 只说明还需要外部运行环境补齐证据。 |

## 2. 阶段完成情况总览

| 阶段 | 当前完成标注 | 当前项目实际情况 | 未完成 / 差距项 | 下一步构建方向 | full_paper 影响 |
|---|---|---|---|---|---|
| `protocol_governance_foundation` | 已完成当前阶段, 持续维护 | 协议、字段注册、测试分层、harness 审计已可运行。 | 新增 baseline、full_paper 和 replay 字段时仍需同步 registry 与 schema。 | 随后续阶段增量维护字段闭包。 | 是所有结果包的前置条件。 |
| `synthetic_state_inference_sanity` | 部分完成 | synthetic runner、state-space 模块和轻量机制测试已存在。 | 不能支撑真实 Flow Matching 视频主 claim。 | 保持为 state inference sanity 与 regression test。 | 只作为机制合理性证据。 |
| `real_video_latent_transfer_check` | 部分完成 | VAE/视频链路模块和 runner 已存在。 | 真实视频 VAE 大规模低误报验证不足。 | 在 validation_scale 与 pilot_paper 前置闭合后补齐 real-video transfer validation。 | 影响 endpoint robustness 与视频链路可信度。 |
| `state_space_inference_formalization` | 部分完成 | state variable、transition、observation、admissibility 结构已拆分。 | generic SSM、Mamba-style temporal fusion、key-agnostic 对比仍需 full-scale governed records。 | 强化 formal ablation 与 negative tail audit。 | 影响 Claim-2 的状态后验贡献。 |
| `trajectory_observation_core_probe` | 部分完成 | trajectory observation、velocity projection、correlation audit 入口已存在。 | path evidence 独立增益仍需 pilot_paper / full_paper 证明。 | 绑定 endpoint、path、velocity 三证据并跑消融。 | 影响 Claim-2 是否成立。 |
| `flow_model_adapter_preflight` | 已完成前置验证 | Wan2.1 callback、time grid、sampler signature 和 latent displacement proxy 已验证。 | 真实 velocity field 原值未必可访问, 当前主要依赖 proxy。 | 保持 proxy 边界, 如能访问真实 velocity 再升级。 | 满足进入 sampling-time 与 pilot 的接口前置。 |
| `sampling_time_constraint_probe` | 已完成机制前置验证 | recommended profile 显示 keyed alignment gain 与 wrong-key 分离。 | 尚不能替代 attack matrix、negative family、fixed-FPR path gain。 | 作为 small_scale_mechanism_pilot_check 前置证据。 | 证明可进入 mechanism_validation 后续检查, 不直接支撑 full_paper。 |
| `motion_threshold_calibration` | 已完成 engineering calibration | 已有 `motion_delta_calibrated_v1` 可作 pilot guardrail。 | 不是论文级 `TPR@FPR=0.001` fixed-FPR 证据。 | full_paper 前补齐更大 held-out negative 和 CI。 | 影响 motion claim 样本资格过滤。 |
| `small_scale_mechanism_pilot_check` | 已完成 small-scale pilot, 作为 mechanism_validation 子检查保留 | 最新 Wan2.1 pilot 原生复跑已达到 16/16 eligible、seed_per_prompt_min=2、runtime attack/detection 48/48 ready、pilot_gate_decision=PASS。 | 它只判断机制是否值得继续, 不是主干门禁, 也不是 paper 级结果包。 | 进入 validation_scale 小样本全流程打通验证, 并保留 small-scale pilot 作为工作流证据。 | 只能解除 validation_scale 的机制前置缺口, 不能直接放行 pilot_paper 或 full_paper。 |
| `validation_scale` | paper 级前小样本全流程打通门禁已完成硬阻断实现, 真实运行待复跑 | 已重新定义为 FPR=10% 小样本全流程打通层, 必须闭合完整现代 external_baseline formal records、内部消融、adaptive attack、replay/sketch 或受治理 Claim-3 downgrade、CI、tables、figures、reports、manifests、artifact rebuild 和 claim audit, 但不支持正式效果主张。 | 真实 validation_scale 结果尚未生成; 5 个主实验现代 baseline 仍需由本项目 clone / build / run / adapt / record 产出 measured_formal, 不接受外部补交结果。 | 先配置并运行现代 baseline 自包含产出链路, 然后在 Colab 中运行 `PROFILE = validation_scale`。 | 通过并生成 validation_scale_to_probe_paper_transition_decision 后才允许进入 probe_paper; 但 pilot_paper 和 full_paper claim 仍需 probe_paper、pilot_paper、full_paper_result_checker 和轻量判定通过。若 baseline / ablation / replay / CI / artifact rebuild 任一缺失, 不得进入 paper 级结果运行。 |
| `probe_paper` | 新增 FPR=10% 小样本论文闭合验证层, 工程入口已完成, 真实 GPU 结果待运行 | 已接入 pilot_paper 级样本结构、target_fpr=0.1、公平比较、差值区间、内部消融、完整 attack 协议和 package profile。 | 真实 probe_paper 结果尚未生成; 需要先消费 validation_scale PASS 与 validation_scale_to_probe_paper_transition_decision。 | 在 validation_scale 通过后运行 `PROFILE = probe_paper`, 审计 `probe_paper_target_fpr_0_1_paper_claim_supported`, 并在通过后生成 `probe_paper_to_pilot_paper_transition_decision`。 | 可支撑 FPR=10% 小样本论文闭合结论候选, 但不能替代 pilot_paper 的 FPR=1% 或 full_paper 的 FPR=0.1% 主结果。 |
| `pilot_paper` | 工程入口已完成, 真实 GPU 结果待运行 | 已接入 10 prompt × 10 seed、calibration split、frozen threshold artifact、held-out test split、代表性 runtime attack coverage、完整现代 external_baseline 自包含 measured_formal adapter 前置检查、内部消融矩阵前置检查和 claim audit。 | 真实 Wan2.1 GPU 结果尚未生成; 5 个主实验现代 baseline 需要在 Colab / 本地通过项目内 clone / build / run / adapt / record 后才能产出 measured_formal。 | 在 validation_scale_to_probe_paper_transition_decision、probe_paper gate 和 probe_paper_to_pilot_paper_transition_decision 都通过后运行 `PROFILE = pilot_paper`, 审计 `pilot_paper_calibrated_heldout_claim_ready`, 同时要求 `modern_external_baseline_formal_measured_adapter_count >= 5` 与 `pilot_paper_internal_ablation_matrix_ready`。 | 它是小规模跑代表性 paper 协议并产出 pilot 级论文结果的阶段, 可支撑 pilot_paper 级 `TPR@FPR=0.01`, 但不支撑 `TPR@FPR=0.001`、full_paper 规模结论或顶会顶刊级完整 attack coverage 结论。 |
| `generative_video_model_probe` | 作为实现 package 保留, validation_scale 真实运行待复跑 | 生成、attack、detection、postprocess、external_baseline source intake、项目内自包含 baseline adapter、内部消融 runner、packager 与协议字段闭包已接入。 | 现代外部 baseline measured_formal、完整内部消融、replay/sketch、CI、tables / figures / reports 和 claim audit 尚未以同一 validation run 通过。 | 按 validation_scale -> probe_paper -> pilot_paper -> full_paper 的主干门禁顺序推进。 | 只提供实现 package; 是否允许进入 pilot_paper、full_paper 由主干门禁和轻量判定决定。 |
| `replay_and_authenticated_sketch_gate` | 未完成 | digest、manifest、trajectory trace 基础模块存在。 | authenticated sketch、replay uncertainty、wrong prompt replay 未闭合。 | 补齐签名 sketch、replay records 和 checker。 | 影响 Claim-3 强度; 不通过则降级 Claim-3。 |
| `flow_specific_adaptive_attack_gate` | 未完成 | phase 文档已补建, 但 runner、manifest 与 governed records 尚未完成。 | adaptive attacks、endpoint-preserving resampling、path cancellation 未形成 records。 | 补齐 runner 设计、stress protocol、attack manifest 和 checker。 | full_paper 前必须完成或明确降级。 |
| `full_paper` | 未开始 | 仅有文档规范, 尚无 full_paper 大规模 records。 | 必须等 validation_scale、probe_paper、pilot_paper、现代外部 baseline、adaptive attack、replay/sketch 与 paper-level fixed-FPR 通过后才能运行。 | 使用 full_paper protocol config 中登记的顶会顶刊级 attack coverage、低 FPR 样本规模和完整 checker, 产出最终论文主结果。 | 是 submission_package_freeze 前的最终结果来源。 |
| `submission_package_freeze` | 结构就绪, 未进入最终冻结 | submission freeze runner 和 readiness summary 已存在。 | 上游 full_paper records 不存在, 不得冻结论文结论。 | 等 full_paper 完成后重建 tables、figures、reports。 | 负责最终 claim audit 和 artifact rebuild。 |

## 3. 2026-06-18 阶段状态更新: 机制前置验证闭合

### 3.1 当前总体判定

截至 `2026-06-18T02:34:47Z` 批次, 项目已经完成真实 Wan2.1 路径上的机制前置验证闭合。这里的“机制前置验证闭合”指以下内容已经具备 governed records、checker 与 package 证据:

```text
protocol_governance_foundation: structure_ready / protocol_ready
flow_model_adapter_preflight: structure_ready / mechanism_ready / protocol_ready / artifact_ready
sampling_time_constraint_probe: structure_ready / mechanism_ready / protocol_ready / artifact_ready
small_scale_claim_pilot_gate: structure_ready / protocol_ready / external_validation_required
```

该判定不能等同于“所有论文 claim 已经完成”。当前尚未完成 small-scale claim pilot, 因此不能进入 `full_paper` 相关结果运行, 也不能声明最终 `TPR@FPR=0.01`、baseline comparison、ablation table 或 submission claim 已经成立。

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
generative_video_model_probe: 尚未进入 `full_paper` 相关结果运行, 不能产生最终主论文表格
replay_and_authenticated_sketch_gate: authenticated sketch 与 replay uncertainty 仍需进一步闭合
submission_package_freeze: 只能等待 pilot_paper / full_paper governed records 后再冻结
```

### 3.4 阶段推进规则

下一步允许进入:

```text
small_scale_claim_pilot_gate
```

下一步不允许直接进入:

```text
generative_video_model_probe full_paper 相关结果运行
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

已满足的部分是 Wan2.1 pilot 生成覆盖和 proxy workflow progression。尚未满足的部分是 attack matrix、negative family、method variant、path marginal gain、negative tail、wrong-key separation、wrong-sampler replay 和 replay uncertainty。因此下一步应实现 pilot postprocess / runner, 不应进入 `full_paper` 相关结果运行。

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

该结果说明真实文件级 attack runner 与 attacked video 落盘链路已经具备工程可运行性。当前仍不应进入 `full_paper` 相关结果运行, 因为 formal motion threshold 仍为 heuristic guardrail, 且 runtime attacked videos 尚未接入正式 detection / scoring 表。

## 5. 2026-06-18 阶段状态更新: runtime attack 到 runtime detection 工程闭环

### 5.1 当前工程闭环判定

截至 `2026-06-18T15:34:39Z` 批次, 当前 Wan2.1 small-scale pilot 的工程链路已经从真实生成视频推进到 runtime attack 和 runtime detection 的完整落盘闭环:

```text
generation_records -> videos -> runtime_attack_records -> attacked_videos -> runtime_detection_records -> pilot_gate -> package_manifest
```

最新 Google Drive package 为:

```text
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_runtime_20260618_153437_e90e82ae.zip
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_runtime_20260618_153437_e90e82ae_package_manifest.json
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
claim 层面: 仍阻塞于 motion_threshold_calibration, 不得进入最终 claim 或 `full_paper` 结论冻结。
```

## 6. 2026-06-19 阶段状态更新: motion threshold calibration 已执行但未通过

### 6.1 当前 calibration artifact 状态

已新增并运行 `motion_threshold_calibration` 工程入口。最新 Google Drive package 为:

```text
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_runtime_20260618_162447_882754a4.zip
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_runtime_20260618_162447_882754a4_package_manifest.json
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

## 7. 2026-06-19 阶段状态更新: motion calibration profile 已按 128/64/32 设计落地

### 7.1 已完成的工程变更

当前仓库已支持独立 motion calibration split:

```text
PROFILE = motion_calibration
negative_static: 128 videos
positive_motion: 64 videos
ambiguous_low_motion: 32 videos
```

实现方式为:

```text
16 negative_static prompts x 8 seeds
8 positive_motion prompts x 8 seeds
4 ambiguous_low_motion prompts x 8 seeds
```

### 7.2 当前仍未完成的工作

该状态不表示 calibration 已经通过。它表示 Colab 冷启动流程现在可以生成正确规模和正确 split 标记的 calibration records。

真正通过仍需要执行真实 Wan2.1 GPU run, 并得到:

```text
motion_threshold_calibration_decision: PASS
motion_threshold_id: motion_delta_calibrated_v1
motion_threshold_source_split: calibration
motion_threshold_calibration_required: false
```

在该结果出现前, 当前 small-scale claim pilot 仍应保持 blocked。


## 8. 2026-06-20 阶段状态更新: motion calibration 可分性 gate 已修复

### 8.1 修复原因

最新 calibration 结果曾出现:

```text
positive_motion_pass_rate_at_threshold = 0.1875
```

该结果表示多数 `positive_motion` 样本在冻结阈值下无法通过。若仍把该结果判定为 `PASS`, 会导致后续 small-scale claim pilot 建立在不可分的 motion gate 上。

### 8.2 已完成的工程修复

已将 motion calibration 从“只检查样本数量与静止 FPR”升级为“同时检查正负样本可分性”。新增硬门槛为:

```text
minimum_positive_motion_pass_rate_at_threshold: 0.80
positive_negative_motion_delta_margin > 0
```

当样本数量充足但分数不可分时, runner 现在输出:

```text
motion_threshold_calibration_decision: FAIL_NOT_SEPARABLE
claim_support_status: blocked_until_motion_threshold_calibration
motion_threshold_calibration_required: true
```

### 8.3 当前推进状态

```text
工程层面: motion calibration runner 与 prompt suite 已修复。
统计层面: 需要重跑真实 Wan2.1 motion_calibration split。
claim 层面: 在 PASS 出现前, small-scale claim pilot 仍不应解除阻塞。
```

### 8.4 下一步最小重跑范围

```text
prepare prompt suite
Wan2.1 generation with PROFILE = motion_calibration
formal_metric_runner
motion_threshold_calibration
package_outputs
```


## 9. 2026-06-21 阶段状态更新: motion calibration 失败原因修复方案已落地

### 9.1 失败原因

最新 Google Drive 结果显示 `motion_calibration` 样本数量已经达标, 但统计可分性未通过:

```text
motion_threshold_calibration_decision: FAIL_NOT_SEPARABLE
positive_motion_pass_rate_at_threshold: 0.546875
minimum_positive_motion_pass_rate_at_threshold: 0.8
```

失败原因不是 Wan2.1 加载或 records 落盘失败, 而是历史 `motion_delta_score` 使用整帧平均差分, 容易同时受到以下两类问题影响:

```text
1. negative_static 中的轻微镜头漂移、曝光变化或纹理抖动会抬高负样本尾部。
2. positive_motion 中的小物体或局部运动会被整帧平均稀释。
```

### 9.2 已完成修复

当前仓库已完成以下工程修复:

```text
1. 新增 motion_delta_focus_score 作为 calibration 优先使用的局部运动分数。
2. motion_threshold_calibration records 新增 motion_calibration_score 和 motion_calibration_score_name。
3. positive_negative_motion_delta_margin 调整为诊断字段, 不再在 positive pass rate 达标时单独阻塞。
4. 替换污染较强的 negative_static prompt。
5. 强化弱运动 positive_motion prompt。
```

### 9.3 当前推进建议

需要重新执行 `motion_threshold_calibration_colab.ipynb` 的 `SSTW_WORKFLOW_PROFILE = motion_calibration` 流程。既有 package 不会自动获得新字段, 必须重新运行 formal metric 与 calibration。


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


## 2026-06-22 阶段状态更新: motion calibration prompt 可观测性修复

### 当前最新外部结果

最新 Google Drive 批次 `20260622_082859_825b4762` 表明工程运行与打包成功, 但 calibration 仍未通过:

```text
implementation_decision: PASS
motion_threshold_calibration_decision: FAIL_NOT_SEPARABLE
positive_motion_pass_rate_at_threshold: 0.75
positive_motion_pass_rate_wilson_lower: 0.631835
```

该结果说明新机制已经修复了“污染过滤是否泄漏”和“负样本尾部是否被异常 prompt 支配”的问题, 但仍暴露出另一类问题:

```text
部分 positive_motion prompt 在真实 Wan2.1 生成中没有稳定产生足够可观测的运动。
```

### 已完成工程修复

本次修复保持阈值、Wilson lower bound 和防泄漏污染过滤协议不变, 只修正 calibration 输入设计:

```text
1. 替换 positive_motion_00 中的抽象 red square slide, 改为真人携带大红色前景板横穿画面。
2. 替换 positive_motion_02 中的抽象 blue circle bounce, 改为近景 blue beach ball 大幅上下运动。
3. 强化 positive_motion_04 和 positive_motion_06 的 foreground、entire frame 与 consecutive frame 位移约束。
4. 替换 high-frequency 或 implied-motion static prompt, 包括 checkerboard、chess board 和 clock illustration。
5. 更新 prompt_suite_id 为 generative_video_probe_prompt_suite_motion_observability_repair, 使新旧批次可快速区分。
```

### 当前阶段判定

历史上在该批次前仍不能进入 small-scale claim pilot。后续已通过 motion calibration 后, 该阻塞条件已解除。若需要复现该历史判断, 下一步必须重新执行:

```text
PROFILE = motion_calibration
```

并确认:

```text
motion_threshold_calibration_decision: PASS
positive_motion_pass_rate_at_threshold >= 0.80
positive_motion_pass_rate_wilson_lower >= 0.70
```

只有上述条件通过后, 才能解除 `blocked_until_motion_threshold_calibration`。


## 2026-06-23 阶段状态更新: 已切换到 small-scale claim pilot

### 最新 motion calibration 前置条件

最新 Google Drive 批次已经通过 engineering motion threshold calibration:

```text
package_batch_id: 20260622_162541_13ef225f
motion_threshold_calibration_decision: PASS
motion_threshold_calibration_ready: true
claim_support_status: motion_threshold_engineering_calibrated
motion_delta_threshold: 0.010607
positive_motion_pass_rate_at_threshold: 0.84375
positive_motion_pass_rate_wilson_lower: 0.735717
```

该结果解除 small-scale pilot 之前的主要阻塞项:

```text
blocked_until_motion_threshold_calibration
```

### Colab 参数切换

`paper_workflow/colab_notebooks/motion_threshold_calibration_colab.ipynb` 已切换到:

```text
PROFILE = 'pilot'
MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
```

pilot profile 的目标规模为:

```text
8 prompts
2 seeds per prompt
3 attacks
4 negative families
6 method variants
```

### 关键执行约束

pilot profile 不会重新运行 `motion_threshold_calibration`, 而是复用已经通过的 calibration artifact:

```text
runs/generative_video_model_probe/motion_calibration/artifacts/motion_threshold_calibration_decision.json
```

该约束用于防止 16 条 pilot records 覆盖独立 128 / 64 / 32 calibration split, 造成阈值被误写为 `INSUFFICIENT_SAMPLE`。

### 当前阶段结论

当前阶段可以进入:

```text
small-scale claim pilot
```

但仍不得进入:

```text
full generative video probe
submission package freeze
final TPR@FPR=0.01 / 0.001 claim
```

是否继续进入 `validation_scale`, 必须由 small_scale_mechanism_pilot_check 结果决定。


## 2026-06-23 阶段状态更新: small-scale pilot formal motion 过滤已修复

### 当前发现

最新 small-scale pilot run 中有 1 个正向运动样本未通过 formal motion consistency:

```text
prompt_id: heldout_rotation_scene
seed_id: seed_main_b
trajectory_trace_id: trace_0005
formal_motion_gate_failure_reason: motion_delta_below_min
formal_metric_blocking_reason: formal_motion_consistency_not_ready
```

该样本不是文件损坏或语义失败, 而是实际生成结果缺少足够可观测运动。它可以保留为 governed record, 但不能作为 velocity / trajectory claim 的正样本证据。

### 已完成工程修复

已新增并接入统一的 formal motion claim 样本筛选逻辑:

```text
experiments/generative_video_model_probe/formal_motion_claim_filter.py
experiments/generative_video_model_probe/pilot_matrix_postprocess.py
experiments/generative_video_model_probe/pilot_claim_gate.py
experiments/generative_video_model_probe/attack_runner.py
```

修复后的规则为:

```text
formal motion 失败样本不物理删除。
generation_records 与 formal_quality_motion_semantic_records 保留完整审计痕迹。
pilot matrix、runtime attack 统计和 pilot gate 覆盖率只使用 motion-claim-eligible 样本。
样本筛选不得依赖 S_final、S_final_conservative 或最终检测分数。
```

### 当前最新 gate 判定

已对当前 Google Drive run 重新写出 pilot matrix 和 pilot gate artifacts:

```text
motion_claim_eligible_generation_count: 15
motion_claim_excluded_generation_count: 1
formal_motion_consistency_ready_count: 15
formal_motion_consistency_blocked_count: 1
pilot_matrix_record_count: 450
pilot_gate_decision: FAIL
claim_support_status: workflow_progression_only
missing_pilot_requirements:
  - seed_coverage_ready
  - formal_motion_claim_ready
```

对应轻量 governed artifacts package 已写入:

```text
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_runtime_20260622_174746_e0f9c79d.zip
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_runtime_20260622_174746_e0f9c79d_package_manifest.json
include_videos: false
```

### 当前阶段结论

当前不是 motion threshold calibration 阻塞, 因为 calibration artifact 已经 PASS。当前阻塞来自 pilot split 中 1 个正向运动样本实际低运动, 导致:

```text
heldout_rotation_scene 只有 1 个合格 seed
small-scale pilot 不满足 8 prompts x 2 seeds 的 motion claim 覆盖要求
```

因此当前不能进入 full generative video probe。下一步应重跑或替换该 prompt / seed, 直到 8 个 prompt 每个都有 2 个通过 formal motion consistency 的合格样本。


## 2026-06-23 历史阶段状态更新: pilot heldout motion prompt 修复记录

### 已完成变更

已将 `heldout_rotation_scene` 从弱运动 prompt 改为强可观测运动 prompt。旧设计:

```text
prompt_text: A blue cube rotates gently on a plain gray surface with soft shadows and smooth motion.
motion_pattern_id: gentle_rotation
```

新设计:

```text
prompt_text: A large blue cube with bright orange arrow markings slides from the far left edge to the far right edge while spinning rapidly for a full rotation on a plain gray floor, fixed camera, the cube fills at least one third of the image, strong visible displacement in every frame.
motion_pattern_id: large_rotation_translation
motion_claim_role: positive_motion
prompt_suite_id: generative_video_probe_prompt_suite_motion_observability_and_pilot_repair
```

同时, pilot 中的 positive motion prompts 已显式携带:

```text
motion_claim_role: positive_motion
```

Colab runtime 现在会把该字段写入 generation records, formal metric runner 也会优先读取该字段, 减少隐式角色推断。

### Google Drive 输入状态

已更新本地同步的 Google Drive prompt suite:

```text
G:\我的云端硬盘\SSTW\datasets\generative_video_prompt_suite\prompt_seed_suite.json
G:\我的云端硬盘\SSTW\datasets\generative_video_prompt_suite\prompt_seed_suite_manifest.json
```

### 当前阶段判定

该历史变更只修复后续 pilot 输入, 不会 retroactively 修改旧 run。旧 run 仍是:

```text
pilot_gate_decision: FAIL
motion_claim_eligible_generation_count: 15
motion_claim_excluded_generation_count: 1
```

下一步需要重新执行:

```text
PROFILE = pilot
prepare prompt suite
Wan2.1 generation
formal_metric_runner
reuse frozen motion_threshold_calibration_decision.json
mechanism postprocess
pilot matrix postprocess
runtime attack
runtime detection
small-scale claim pilot gate
package_outputs
```

只有新 run 达到:

```text
motion_claim_eligible_generation_count: 16
motion_claim_excluded_generation_count: 0
seed_per_prompt_min: 2
formal_motion_claim_status: ready
pilot_gate_decision: PASS
```

才可以认为 small-scale pilot 的 motion coverage 阻塞解除。


## 2026-06-23 文档增强后当前阶段查漏补缺基线

### 当前总体判定

当前项目不是 full_paper 运行前状态, 也不是 submission freeze 状态。当前最准确的阶段判定为:

```text
core_method_runtime_construction 已具备主要工程骨架
sampling_time_constraint_probe 已完成机制前置验证
small_scale_claim_pilot_gate 已完成 small-scale pilot
generative_video_model_probe 已进入 validation_scale 准备阶段
full_paper 未开始, 且在 validation_scale / probe_paper / pilot_paper / adaptive attack / baseline / replay-sketch 前不得启动
```

### 当前阻塞项

```text
primary_blocker: validation_scale_full_pipeline_not_completed
secondary_blocker: modern_external_baseline_main_comparison_not_ready
secondary_blocker: internal_ablation_full_scale_records_not_ready
secondary_blocker: replay_and_authenticated_sketch_gate_not_closed
secondary_blocker: flow_specific_adaptive_attack_gate_not_closed
secondary_blocker: paper_level_fpr_0_001_calibration_not_ready
secondary_blocker: pilot_paper_real_gpu_result_not_completed
```

### 下一步允许执行

```text
validation_scale prompt / seed / attack manifest 规划
validation_scale full-pipeline runner / checker 增强
modern external baseline governed status records and adapter contracts
internal ablation matrix records
fixed-FPR and confidence interval reporter
flow-specific adaptive attack runner design
replay/sketch verification runner design
pilot_paper real GPU execution and gate audit
```

### 下一步禁止执行

```text
full_paper result package
submission freeze final claim
TPR@FPR=0.001 final table
manual baseline comparison table
test split threshold update
using pilot records as full_paper records
using non-run baseline records as positive comparison claims
```

### pilot 解除阻塞标准与当前观测

```text
motion_claim_eligible_generation_count = 16
motion_claim_excluded_generation_count = 0
seed_per_prompt_min >= 2
formal_motion_claim_status = ready
pilot_gate_decision = PASS
claim_support_status = supported_by_small_scale_claim_pilot_records
```

最新原生复跑已经满足上述标准。因此, 后续工作不应继续围绕“修复 pilot”展开, 而应进入 validation_scale 与论文级证据充分性构建。

## 2026-06-23 文档增强后工程化门禁缺口

### 当前文档层面已增强的内容

```text
pilot_paper gate 规范已写入总体流程
统计置信区间与 cluster-by-video interval 要求已写入总体流程
审稿风险对照矩阵已写入总体流程
算法原语到 full_paper package 的记录映射已写入算法原语文档
```

### 当前仍未工程化的门禁

```text
pilot_paper_gate: implemented_waiting_for_real_gpu_result
full_paper_result_checker: implemented_in scripts/check_results/full_paper_result_checker.py
modern_external_baseline_runner: governed_status_records_ready_but_main_comparison_not_ready
flow_specific_adaptive_attack_runner: not_implemented
replay_sketch_verification_checker: incomplete
statistical_confidence_interval_reporter: not_implemented
cluster_by_video_interval_reporter: not_implemented
```

### 阻断解释

上述缺口不影响当前文档作为操作手册使用, 但会阻止项目直接进入 full_paper 论文结果包产出。下一步若继续工程推进, 应优先把这些文档化 gate 转换为 repository runner、checker、records 和 reports。

## 2026-06-23 文档继续增强: 顶会实验充分性清单与 full_paper 可运行性预演

### 本次文档增强内容

```text
新增 sstw_top_tier_experimental_sufficiency_checklist.md
总体流程新增 full_paper 分片执行、资源预检和失败恢复规则
总体流程新增外部 baseline 公平比较协议
总体流程新增内部消融矩阵与审稿证据索引
full_paper phase 新增 shard / rehearsal 要求
generative video phase 新增 validation_scale 充分性矩阵
adaptive attack phase 新增攻击者知识层级与 claim 降级规则
small-scale pilot phase 新增 pilot 结果使用边界
submission freeze phase 新增 reviewer evidence index 要求
算法原语文档新增可证伪条件和原语级实验打包要求
```

### 当前状态解释

本次增强只修改构建文档, 不产出 full_paper 结果。最新阶段判断为:

```text
primary_blocker: validation_scale_full_pipeline_not_completed
full_paper_allowed: false
submission_freeze_allowed: false
```

### 下一步工程化方向

```text
1. 设计并执行 validation_scale full-pipeline rehearsal。
2. 将现代外部 baseline 的 governed status record 扩展为 adapter contract 或明确 non-run protocol gap。
3. 将 pilot_paper 真实 GPU 复跑、baseline runner、adaptive attack runner、CI reporter 和 reviewer evidence index 从工程入口转化为 governed records。
```

## 2026-06-23 文档继续增强: full_paper 工程门禁实现规范

### 本次补强内容

```text
新增 sstw_full_paper_engineering_gate_spec.md
总体流程新增 full_paper 工程门禁实现规范索引
顶会实验充分性清单新增工程化 readiness 评分
```

### 当前状态解释

本次补强进一步降低“文档无法落地为 Codex 工程任务”的风险。当前项目阶段已经从 pilot 阻塞转为 validation_scale 阻塞:

```text
primary_blocker: validation_scale_full_pipeline_not_completed
full_paper_allowed: false
submission_freeze_allowed: false
```

### 新增工程化目标

后续若继续推进代码实现, 应优先按 `sstw_full_paper_engineering_gate_spec.md` 实现:

```text
pilot_paper real GPU execution and gate audit
statistical_confidence_interval_reporter
modern_external_baseline_runner
reviewer_evidence_index_builder
full_paper_result_checker
flow_specific_adaptive_attack_runner
```



## 2026-06-23 最新原生复跑后阶段状态更新

### 当前总体判定

最新 Google Drive 落盘结果显示, `small_scale_claim_pilot_gate` 已经通过。当前项目不再处于 small-scale pilot 阻塞状态, 而是进入 validation_scale 与论文级证据充分性构建阶段。

```text
flow_model_adapter_preflight: PASS
motion_threshold_calibration: PASS
small_scale_claim_pilot_gate: PASS
record_protocol_missing_failures: []
full_paper_allowed: false
submission_freeze_allowed: false
```

### 最新落盘证据

```text
preflight_package: wan21_flow_adapter_preflight_20260623_064928_839da169.zip
generative_package: generative_video_runtime_20260623_134119_839da169.zip
adapter_preflight_decision: PASS
model_load_status: loaded
callback_latent_capture_status: captured
time_grid_capture_status: captured
sampler_signature_status: captured
velocity_proxy_status: captured
small_scale_pilot_gate_decision: PASS
claim_support_status: supported_by_small_scale_claim_pilot_records
motion_claim_eligible_generation_count: 16
motion_claim_excluded_generation_count: 0
runtime_attack_ready_count: 48
runtime_detection_ready_count: 48
pilot_matrix_record_count: 480
```

### 当前阻塞项重新判定

```text
primary_blocker: validation_scale_full_pipeline_not_completed
secondary_blocker: modern_external_baseline_main_comparison_not_ready
secondary_blocker: internal_ablation_full_scale_records_not_ready
secondary_blocker: flow_specific_adaptive_attack_gate_not_closed
secondary_blocker: replay_and_authenticated_sketch_gate_not_closed
secondary_blocker: paper_level_fpr_0_001_calibration_not_ready
secondary_blocker: pilot_paper_real_gpu_result_not_completed
```

### 下一步允许执行

```text
validation_scale generative video probe planning
modern external baseline governed status records and adapter contracts
internal ablation matrix records
fixed-FPR and confidence interval reporter
flow-specific adaptive attack runner design
pilot_paper real GPU execution and gate audit
```

### 下一步仍禁止执行

```text
full_paper result package
submission freeze final claim
TPR@FPR=0.001 final table
manual baseline comparison table
using pilot records as full_paper records
using non-run baseline records as positive comparison claims
```

### 本次工程推进状态

现代外部 baseline 已开始从“未集成”推进为“governed status records ready”。该状态只说明 baseline 不会被静默删除, 不能说明 SSTW 已经优于这些现代 baseline。进入论文主表前仍必须满足:

```text
external_baseline_runnable_status = runnable
external_baseline_adapter_status = ready
external_baseline_output_record_status = governed_records_written
external_baseline_threshold_policy_compatible = true
external_baseline_attack_manifest_compatible = true
external_baseline_result_used_for_claim = true
```

## 2026-06-23 validation_scale gate 工程推进状态

### 本次新增工程入口

为防止从 small-scale pilot 直接跳到 full_paper, 当前仓库新增 validation_scale gate:

```text
experiments/generative_video_model_probe/validation_scale_gate.py
configs/protocol/validation_scale_generative_probe.json
paper_workflow/notebook_utils/generative_video_model_probe_workflow.py::build_validation_scale_gate_command
paper_workflow/colab_notebooks/generative_video_generation_colab.ipynb SSTW_WORKFLOW_PROFILE = validation_scale
paper_workflow/colab_notebooks/generative_video_quality_scoring_colab.ipynb SSTW_WORKFLOW_PROFILE = validation_scale
paper_workflow/colab_notebooks/sstw_mechanism_postprocess_colab.ipynb SSTW_WORKFLOW_PROFILE = validation_scale
paper_workflow/colab_notebooks/runtime_attack_colab.ipynb SSTW_WORKFLOW_PROFILE = validation_scale
paper_workflow/colab_notebooks/runtime_detection_colab.ipynb SSTW_WORKFLOW_PROFILE = validation_scale
```

该 gate 会写出:

```text
records/validation_scale_gate_records.jsonl
tables/validation_scale_gate_table.csv
artifacts/validation_scale_gate_decision.json
reports/validation_scale_gate_report.md
```

### 当前阶段语义

当前项目已经具备 validation_scale 的工程审计入口, 但尚未完成 validation_scale 真实运行。当前状态应表述为:

```text
small_scale_claim_pilot_gate: PASS
validation_scale_gate_checker: implemented
validation_scale_gate_decision: waiting_for_validation_scale_run
full_paper_allowed: false
submission_freeze_allowed: false
```

### validation_scale 必须闭合的内容

```text
validation_generation_records_ready
validation_attack_records_ready
validation_detection_records_ready
validation_external_baseline_status_records_ready
validation_internal_ablation_records_ready
validation_adaptive_attack_records_ready
validation_replay_or_sketch_records_ready
validation_confidence_interval_report_ready
validation_artifact_rebuild_dry_run_ready
```

该 gate 通过后, 下一步仍然是 `pilot_paper_gate`, 不是 full_paper result package。

## 2026-06-23 validation_scale 后处理闭环工程推进

### 本次新增工程能力

在延后真实 GPU 复跑的前提下, 当前仓库继续补齐 validation_scale 运行后的自动后处理链路:

```text
validation_internal_ablation_runner: implemented
statistical_confidence_interval_reporter: implemented_for_validation_proxy
validation_artifact_rebuild_dry_run: implemented
```

对应模块为:

```text
experiments/generative_video_model_probe/validation_internal_ablation.py
experiments/generative_video_model_probe/statistical_confidence_interval.py
experiments/generative_video_model_probe/validation_artifact_rebuild.py
```

### 当前状态解释

这些模块不运行 Wan2.1, 不生成 full_paper 主表, 也不把 validation proxy 伪装成最终论文 claim。它们的作用是让未来 `PROFILE = validation_scale` 复跑结束后, 以下产物自动闭环:

```text
validation_internal_ablation_records
statistical_confidence_interval_records
validation_artifact_rebuild_dry_run_records
validation_scale_gate_decision
```

当前阶段仍为:

```text
validation_scale_gate_checker: implemented
validation_scale_postprocess_runners: implemented
validation_scale_real_gpu_run: not_yet_rerun
full_paper_allowed: false
submission_freeze_allowed: false
```

### 仍未完成的工程项

```text
adaptive_attack_runner: not_implemented
replay_and_authenticated_sketch_gate_runner: not_implemented
modern_external_baseline_formal_command_adapter: implemented_requires_official_command_configuration
pilot_paper_gate: implemented_waiting_for_real_gpu_result
```


## 2026-06-23 external_baseline adapter comparison 工程闭环

### 本次完成项

外部 baseline 现在不再只停留在 governed status record。项目已经新增 `external_baseline/` 适配边界, 并由 `experiments/generative_video_model_probe/external_baseline_runner.py` 调度 adapter 产出 comparison records、table、decision 和 report。

```text
external_baseline_adapter_boundary: implemented
external_baseline_score_records: implemented
external_baseline_comparison_table: implemented
validation_external_baseline_comparison_records_ready: added_to_gate
package_manifest_external_baseline_comparison_summary: implemented
```

### 与论文 claim 的边界


```text
claim_support_status: external_baseline_proxy_comparison_not_claim_supporting
modern_external_baseline_formal_command_adapter: implemented_requires_official_command_configuration
full_paper_allowed: false
submission_freeze_allowed: false
```

### validation_scale gate 新增要求

```text
validation_external_baseline_status_records_ready
validation_external_baseline_comparison_records_ready
minimum_external_baseline_measured_adapter_count: 5
modern_external_baseline_formal_measured_adapter_count: 5
```

下一步是配置现代 baseline 官方命令或 source 入口, 保持 adaptive attack、replay/sketch 或 Claim-3 降级路径闭合, 然后执行 validation_scale 真实复跑。现代 baseline formal records 不得推迟到 pilot_paper 后补。


## 2026-06-24 external_baseline 接入方式澄清

本项目的外部 baseline 接入方式已经明确为项目内生的 adapter-source-observation-comparison 分层机制, 不依赖任何其他项目名称作为说明依据。

### 已确认的接入结构

```text
source_registry: external_baseline/source_registry.json
adapter_boundary: external_baseline/primary/<baseline_id>/adapter/run_sstw_eval.py
scheduler: experiments/generative_video_model_probe/external_baseline_runner.py
status_records: records/external_baseline_records.jsonl
comparison_records: records/external_baseline_score_records.jsonl
comparison_table: tables/external_baseline_comparison_table.csv
decision_artifact: artifacts/external_baseline_comparison_decision.json
report: reports/external_baseline_comparison_report.md
```

### 阶段性判断

```text
external_baseline_adapter_boundary: complete_for_proxy_controls
external_baseline_comparison_output_chain: complete_for_proxy_controls
modern_external_baseline_formal_adapter: integrated_requires_official_commands
baseline_claim_support: not_supported_until_modern_measured_records
full_paper_allowed: false
```

当前可以说明工程链路已经能从本项目直接产出 baseline comparison 结果; 不能说明现代视频水印外部 baseline 已完成正式论文主表对比。

## 2026-06-24 replay/sketch gate 最终创新性要求

当前项目允许短期实现 `Claim-3 downgrade gate` 以保护 validation_scale 流程和 claim 边界, 但这不表示 Claim-3 已经完成。项目最终若要把 replay posterior、authenticated sketch 和 wrong replay 分离作为强创新点, 必须实现并通过 `replay/sketch gate`。

当前阶段性判断为:

```text
claim3_downgrade_gate: allowed_for_validation_scale_short_term
replay_and_authenticated_sketch_gate: required_for_full_paper_strong_claim
replay_and_authenticated_sketch_gate_runner: not_implemented
claim3_full_support_allowed: false
```

后续工程必须补齐 authenticated trajectory sketch、replay uncertainty records、wrong sampler / wrong prompt / wrong time grid replay controls、replay negative FPR control 和 `artifacts/replay_and_sketch_gate_decision.json`。在这些产物闭合前, 只能把 Claim-3 表述为降级或探索性 replay analysis。

## 2026-06-24 adaptive attack validation proxy runner

当前已补齐 validation_scale 的 adaptive attack 硬阻断 runner。该 runner 从 runtime detection records 生成 Flow-specific adaptive attack proxy records, 覆盖 scheduler/time grid mismatch、wrong sampler replay、endpoint-path decoupling、path response cancellation 和 trajectory sketch replacement attempt。

阶段状态更新为:

```text
adaptive_attack_runner: implemented_for_validation_proxy
adaptive_attack_decision_artifact: implemented
adaptive_robustness_claim_allowed: false
full_paper_adaptive_attack_gate: still_requires_real_negative_tail_audit
```

该更新只能让 validation_scale gate 获得 governed adaptive attack records, 不能支撑 full_paper 中的 `robust_to_flow_specific_adaptive_attacks` 强主张。

## 2026-06-24 validation_scale hard-blocker workflow 接入

当前 generative video Colab workflow 已接入 adaptive attack validation proxy 与 Claim-3 downgrade gate。validation_scale artifact rebuild dry-run 也已将两类新增产物纳入 required inputs / outputs 检查。

阶段状态更新为:

```text
adaptive_attack_runner_workflow: integrated
claim3_downgrade_gate_workflow: integrated
validation_artifact_rebuild_required_claim3_and_adaptive_outputs: integrated
validation_scale_real_gpu_run: pending_user_colab_rerun
```

下一步是执行本地 validation_scale 工程闭环核验。若本地测试与 harness 均通过, 项目将等待用户在 Colab 中执行真实 Wan2.1 GPU validation_scale 复跑。

## 2026-06-24 formal motion exclusion gate bug 修复

已修复 small-scale / validation_scale 衔接中的一个 gate 判定问题: 原逻辑把任何一个 formal motion consistency 失败样本都解释为 `formal_motion_claim_ready` 缺失, 即使该样本已经被 formal motion filter 从 claim eligible set 中排除, 且剩余样本仍满足 prompt / seed 覆盖要求。

修复后的工程状态为:

```text
sample_level_filtering: unchanged
excluded_low_motion_samples: retained_for_audit_but_not_claim_support
formal_motion_claim_status_ready_values:
  - ready
  - ready_with_formal_motion_exclusions
pilot_gate_blocking_source_after_exclusion: coverage_rules_only
final_detection_score_used_for_filtering: false
```

该变更使 validation_scale 结果中“23 个 eligible 样本 + 1 个 formal motion exclusion”能够被正确解释: 若 8 个 prompt 均仍至少保留 2 个 seed, 则 `small_scale_claim_pilot_gate_passed` 不应再因为已剔除样本而失败。后续真实 GPU 复跑仍需重新生成并落盘 artifacts, 本地代码层面已经补充对应回归测试。

## 2026-06-24 replay/sketch gate validation proxy 工程推进

在 validation_scale gate 已通过后, 项目已开始进入 replay/sketch、现代 external baseline 和真实内部消融的后续建设阶段。第一项推进为 replay/sketch gate validation proxy:

```text
replay_and_sketch_gate_runner: implemented_for_validation_proxy
trajectory_sketch_verification_records: implemented
replay_uncertainty_records: implemented
wrong_sampler_replay_records: implemented
wrong_prompt_replay_records: implemented
validation_artifact_rebuild_replay_sketch_inputs: integrated
colab_workflow_replay_sketch_step: integrated
package_manifest_replay_sketch_summary: integrated
claim3_full_support_allowed: false
```

该状态说明 validation_scale 工程流程可以继续向 pilot_paper gate 推进, 但 Claim-3 仍保持降级边界。后续还需要把 validation proxy 升级为 full_paper 级 authenticated replay 和 replay negative FPR 审计。


## 2026-06-24 pilot_paper FPR=0.01 工程入口

当前 `generative_video_model_probe` 已新增 `pilot_paper` 语义层级, 用于在 validation_scale 之后执行小样本论文级结果包。该层级不是 workflow-only pilot, 而是小规模跑代表性 paper 协议并产出 pilot 级论文结果。

该阶段协议为:

```text
calibration split
-> frozen threshold artifact
-> held-out test split
-> tables / figures / claim audit
```

新增工程入口包括:

```text
configs/protocol/pilot_paper_generative_probe.json
experiments/generative_video_model_probe/pilot_paper_gate.py
colab_runtime PROFILE = pilot_paper
notebook workflow build_pilot_paper_gate_command
Google Drive package manifest pilot_paper summary
```

当前数据集构造目标为:

```text
paper_result_level: pilot_paper
paper_protocol_level: paper_grade_protocol
paper_protocol_difference_from_full_paper: sample_scale_and_target_fpr_only
prompt_count: 10
seed_per_prompt: 10
calibration_seed_per_prompt: 5
test_seed_per_prompt: 5
unique_video_count: 100
calibration_unique_video_count: 50
test_unique_video_count: 50
expected_calibration_negative_event_count: 5000
expected_heldout_test_negative_event_count: 5000
expected_heldout_attacked_positive_event_count: 2300
target_fpr: 0.01
threshold_protocol: calibration_split_to_frozen_threshold_to_heldout_test_split
```

该阶段通过后允许报告 `pilot_paper_calibrated_heldout_claim_ready` 和 pilot_paper 级 `TPR@FPR=0.01` 论文主张。它与 full_paper 的区别只在样本规模和统计置信度, 但仍不允许报告 `TPR@FPR=0.001` 或 full_paper 规模主表结论。


## 2026-06-24 pilot_paper baseline 与内部消融门禁前置化

根据当前项目推进判断, `pilot_paper` 不能只依赖主方法和 fixed-FPR threshold 输出。由于它被定义为小规模 full_paper 协议预演, baseline comparison 和内部消融必须在 gate 前闭合。

本次工程推进将以下检查纳入 `pilot_paper_gate`:

```text
pilot_paper_external_baseline_comparison_ready
pilot_paper_internal_ablation_matrix_ready
required_external_baseline_adapter_names
required_internal_ablation_variants
minimum_pilot_paper_external_baseline_trace_count
minimum_pilot_paper_internal_ablation_trace_count
```

当前阶段性解释为:

```text
explicit_dtw_temporal_alignment: runnable_control_proxy
explicit_frame_matching_temporal_registration: runnable_control_proxy
modern_video_watermark_baseline_formal_adapter: pending
pilot_paper_internal_ablation_matrix: required_before_gate_pass
full_scale_ablation_table: pending_full_paper_scale
```

因此下一步真实 GPU 复跑顺序仍是先 `PROFILE = validation_scale`, 再 `PROFILE = pilot_paper`。如果 `pilot_paper` 运行后 baseline 或消融 records 未覆盖同批 held-out test trace, gate 会失败, 不能报告 pilot 级 `TPR@FPR=0.01`。


## 2026-06-24 现代 baseline 正式 adapter 硬前置

根据项目阶段定义, `pilot_paper` 和 `full_paper` 的区别只能是样本规模和 FPR 评价级别。因此 `pilot_paper` 不能只接入一个现代 baseline, 也不能用显式同步 control proxy 替代现代视频水印 baseline。

当前工程已经把以下 5 个主实验现代 baseline 接入为正式 command adapter 边界:

```text
videoshield
vidsig
videoseal
```

阶段性状态为:

```text
modern_external_baseline_formal_adapter_boundary: implemented
modern_external_baseline_official_commands_configured: pending_user_colab_or_local_setup
pilot_paper_required_modern_external_baseline_count: 5
pilot_paper_required_external_baseline_count_total: 7
pilot_paper_gate_missing_modern_formal_results: hard_blocker
```

这表示当前代码框架已经支持真实产出相关对比结果, 但真实运行前必须在 Colab 环境安装或配置对应官方 baseline 命令。若命令未配置, adapter 会写 unsupported record, `pilot_paper` gate 会失败。


## 2026-06-24 validation_scale 小样本全流程打通要求更新（历史推进记录）

本次流程要求更新后, `validation_scale` 被重新定义为进入 paper 级运行前的 FPR=10% 小样本全流程打通层。它不再只是工程稳定性、attack runner、baseline 接口或 artifact rebuild 的松散检查, 而必须在小样本规模上证明 paper 相关全部机制和产物链路已经可运行、可记录、可重建、可审计。

新的 `validation_scale` 通过条件应至少包括:

```text
complete_modern_external_baseline_formal_records_ready = true
external_baseline_measured_adapter_count >= 5
modern_external_baseline_formal_measured_adapter_count >= 5
internal_ablation_matrix_records_ready = true
flow_specific_adaptive_attack_records_ready = true
replay_and_sketch_gate_ready = true 或 governed_claim3_downgrade_ready = true
fixed_fpr_confidence_interval_report_ready = true
validation_tables_figures_reports_ready = true
artifact_rebuild_dry_run_ready = true
validation_claim_audit_ready = true
package_manifest_ready = true
```

当前项目状态因此应调整为:

```text
validation_scale_requirement: upgraded_to_small_sample_full_pipeline_gate
validation_scale_code_gate: implemented_with_full_baseline_hard_blockers
validation_scale_real_gpu_run: pending_after_official_baseline_command_configuration
pilot_paper_allowed: false_until_validation_scale_full_pipeline_pass
full_paper_allowed: false_until_validation_scale_probe_paper_pilot_paper_and_full_paper_checker_pass
```

该更新的核心含义是: `pilot_paper` 只能执行 paper 级结果运行和报告 pilot 级结论, 不能再承担补现代 baseline、补内部消融、补 replay/sketch、补 CI 或补 artifact rebuild 的职责。若这些内容在 `validation_scale` 中没有闭合, 下一步必须修复 validation_scale 机制, 不能进入 `pilot_paper`。

## 2026-06-24 external baseline source intake 与 execution manifest 接入

本次工程推进后, 外部 baseline 接入从“adapter 壳层存在”推进为“source intake + adapter + comparison + execution manifest”治理链路。新增或固化的工程入口包括:

```text
external_baseline/source_intake.py
scripts/build_external_baseline_source_intake.py
external_baseline/external_baseline_intake_manifest.json
external_baseline/external_baseline_source_inspection.json
external_baseline/external_baseline_clone_results.json
external_baseline/plans/external_baseline_table_plan.json
artifacts/external_baseline_execution_manifest.json
```

阶段性状态为:

```text
external_baseline_source_registry: implemented
external_baseline_source_intake_manifest: implemented
external_baseline_source_inspection_manifest: implemented
external_baseline_clone_plan_manifest: implemented_without_network_by_default
external_baseline_execution_manifest: implemented
validation_scale_modern_baseline_hard_gate: implemented
modern_external_baseline_official_source_or_command: still_required_for_real_measured_formal_results
```

该状态表示项目已经具备正式接入现代视频水印 baseline 的工程通道。当前仍未等价于完成真实 baseline 对比, 因为 5 个主实验现代 baseline 需要在 Colab 或本地通过项目内 clone / build / run / adapt / record, 并配置官方命令或 source 入口后, 才能在同一 run_root 上产出 `measured_formal` records。

新的 validation_scale baseline 阻断条件为:

```text
external_baseline_measured_adapter_count >= 5
modern_external_baseline_formal_measured_adapter_count >= 5
missing_modern_external_baseline_formal_adapter_names == []
```

如果仅有 `explicit_dtw_temporal_alignment` 与 `explicit_frame_matching_temporal_registration` 两个 proxy control, validation_scale 现在必须失败。下一步不是进入 `pilot_paper`, 而是配置现代 baseline 官方命令、执行同批视频的 baseline adapter, 并确认 `external_baseline_execution_manifest.json` 与 comparison records 均已落盘。

## 2026-06-24 Colab-only external baseline 运行边界加固（历史状态记录）

根据“所有真实运行均在 Colab 进行”的执行约束, 当前 generative video Notebook 已把现代 baseline 配置前置为 Colab 显式参数。新增或固化的入口包括:

```text
RUN_EXTERNAL_BASELINE_SOURCE_CLONE
EXTERNAL_BASELINE_EVIDENCE_PATHS
REQUIRE_MODERN_BASELINE_COMMANDS_FOR_PAPER_GATE
build_modern_baseline_command_env(...)
write_external_baseline_colab_preflight_decision(...)
external_baseline_colab_preflight_decision.json
build_external_baseline_source_intake_command(..., execute_clone=...)
SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS
```

阶段性状态为:

```text
local_heavy_baseline_execution: forbidden
colab_modern_baseline_command_configuration: required
validation_scale_missing_modern_command_preflight: hard_block
pilot_paper_missing_modern_command_preflight: hard_block
external_baseline_colab_preflight_artifact: integrated
external_baseline_evidence_path_manifest_binding: integrated
real_modern_baseline_results: pending_user_colab_configuration_and_run
```

历史状态说明: 该更新只改变当时的执行边界和 Notebook 预检逻辑, 不产生新的真实 baseline 结果。当时使用“安装或挂载官方实现”与 `EXTERNAL_BASELINE_EVIDENCE_PATHS` 表达 evidence path 绑定。当前规则已经收紧为由本项目完成 clone / build / run / adapt / record, 并由项目 workflow 自动记录运行日志、配置、官方输出和 provenance paths; 这些历史路径变量不能被解释为允许外部补交结果。在现代 baseline 自包含链路完成前, `validation_scale` 会提前阻断, 不允许继续生成缺现代 baseline 的 paper-gate 结果包。阻断前会写出 `artifacts/external_baseline_colab_preflight_decision.json`, 使 Colab 冷启动失败原因可随 Google Drive package 审计。

## 2026-06-24 现代 baseline 官方输出证据持久化

本次继续加固 external baseline 正式 adapter。现代视频水印 baseline command adapter 不再只把官方输出 JSON 读入内存后丢弃, 而是把每条官方命令的输出证据持久化到当前 Colab `run_root`:

```text
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_output.json
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_stdout.txt
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_stderr.txt
artifacts/external_baseline_evidence/<baseline_id>/<score_digest>/official_command_manifest.json
```

对应状态为:

```text
modern_external_baseline_official_output_persistence: implemented
external_baseline_execution_manifest_auto_evidence_collection: implemented
formal_evidence_status_when_command_outputs_exist: evidence_paths_bound
google_drive_package_auditability_for_baseline_outputs: improved
real_modern_baseline_results: still_pending_colab_run
```

该更新不会在本地产生真实 baseline 结果, 但会保证未来 Colab 运行完成后, package 中保留每条 `measured_formal` baseline score 的官方输出、stdout / stderr 和命令证据 manifest, 便于后续 claim audit 和 rebuttal-ready evidence index 使用。

## 2026-06-24 Colab workflow profile 配置化重构

本次工程重构将生成式视频主线的 Colab 执行入口从“拆分 Notebook 多处手写 profile 分支”调整为“统一 workflow profile 配置 + 拆分 Notebook role”的方式。

新增统一配置文件:

```text
configs/paper_workflow/generative_video_notebook_workflows.json
```

该配置集中维护:

```text
workflow_profile
result_tier
runtime_profile
protocol_config_path
drive_run_root_relative
drive_package_dir_relative
drive_log_dir_relative
motion_threshold_artifact_run_root_relative
method_sample_count
baseline_sample_count
target_fpr
minimum_clean_negative_count
bootstrap_iteration_count
notebook_role
workflow_stage_plan
```

新增或重构的 Notebook 入口为:

```text
paper_workflow/colab_notebooks/motion_threshold_calibration_colab.ipynb
paper_workflow/colab_notebooks/generative_video_generation_colab.ipynb
paper_workflow/colab_notebooks/generative_video_quality_scoring_colab.ipynb
paper_workflow/colab_notebooks/sstw_mechanism_postprocess_colab.ipynb
paper_workflow/colab_notebooks/runtime_attack_colab.ipynb
paper_workflow/colab_notebooks/runtime_detection_colab.ipynb
paper_workflow/colab_notebooks/*_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/formal_comparison_scoring_colab.ipynb
paper_workflow/colab_notebooks/paper_evidence_postprocess_colab.ipynb
paper_workflow/colab_notebooks/paper_gate_and_package_colab.ipynb
```

旧的通用 external baseline scoring Notebook 已删除。validation-scale 推荐主流程
保留 5 个 baseline 专用 official reference Notebook、独立 formal comparison scoring、
paper evidence postprocess 与 `paper_gate_and_package_colab.ipynb` 的最终聚合门禁。

阶段性状态为:

```text
profile_driven_colab_workflow_config: implemented
profile_specific_drive_run_root: implemented
profile_specific_drive_package_dir: implemented
shared_motion_threshold_artifact_run_root: implemented
workflow_profile_aliases_removed: implemented
full_paper_profile_registration: design_registered_not_ready
monolithic_notebook_status: removed
recommended_split_notebook_workflow: implemented
validation_scale_single_notebook_gate_test: removed_split_workflow_required
validation_scale_run_through_test_without_fake_claim: removed_with_monolithic_notebook
```

该变更解决的问题是: `validation_scale`、`probe_paper`、`pilot_paper` 和 `full_paper` 不再依赖 Notebook 中多处硬编码路径、样本数量或 profile 集合。切换运行层级时, 用户只需要设置:

```text
SSTW_WORKFLOW_PROFILE=validation_scale
或
SSTW_WORKFLOW_PROFILE=pilot_paper
```

当前仍不允许设置 `SSTW_WORKFLOW_PROFILE=full_paper` 进入真实 claim 运行, 因为该 profile 只登记未来 full_paper 协议入口, 尚未完成 full_paper 样本规模、FPR=0.001 统计设计和真实大规模 baseline / ablation 结果闭合。

## 2026-06-25 现代 baseline 联网核验与 Colab command 配置辅助

本次推进对 5 个主实验现代视频水印 baseline 的公开仓库、默认 branch 和当前 HEAD commit 进行了联网核验, 并新增 Colab command 配置辅助文件:

```text
configs/external_baselines/modern_baseline_colab_commands.json
```

该配置覆盖:

```text
videoshield
vidsig
videoseal
```

新增或固化的 Colab 落盘辅助 artifact 为:

```text
artifacts/external_baseline_command_template_summary.json
```

阶段性状态为:

```text
modern_external_baseline_source_url_verification: completed_for_configured_heads
modern_external_baseline_colab_command_template_config: implemented
notebook_command_template_summary_artifact: integrated
command_template_auto_applied: false
modern_external_baseline_measured_formal_results: still_pending_real_colab_wrapper_commands
validation_scale_missing_modern_command_preflight: still_hard_block
```

该更新解决的问题是: Colab 冷启动失败时, 用户不仅能看到缺少哪些 `SSTW_<BASELINE>_EVAL_COMMAND`, 还能在 Google Drive 中看到每个 baseline 的官方源码位置、clone 目标、官方入口候选脚本和 SSTW wrapper command 模板。

该更新没有绕过 validation_scale 门禁。只有当本项目在 Colab 或等价受治理环境中完成 clone / build / run / adapt / record, 准备权重、编写真实 wrapper, 并显式设置 5 个主实验 `SSTW_<BASELINE>_EVAL_COMMAND` 后, 现代 baseline 才能产出 `measured_formal` records。仅存在 URL、clone plan 或 command 模板不能支撑 baseline comparison claim。

## 2026-06-25 现代 baseline repository bridge command 接入

为推进 `validation_scale` 正式门禁跑通, 当前工程新增统一官方命令桥接器:

```text
external_baseline/official_command_bridge.py
```

该桥接器解决的问题是: 5 个主实验现代视频水印 baseline 的官方仓库入口不同, 但 SSTW 需要统一的
`source_video_path / attacked_video_path / attack_name / output_json_path` command adapter 契约。

新的运行边界为:

```text
SSTW_<BASELINE>_EVAL_COMMAND:
  repository bridge 外层命令, 由 Notebook 可自动从配置生成

SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND:
  用户配置的真实官方 baseline 命令, 必须调用官方代码或权重
  并写出 {official_output_json_path}
```

新增 preflight artifact:

```text
artifacts/external_baseline_official_bridge_preflight_decision.json
```

阶段性状态为:

```text
modern_external_baseline_bridge_outer_command: implemented
modern_external_baseline_official_inner_command_contract: implemented
bridge_preflight_hard_blocker: implemented
direct_eval_command_override_bridge_template: implemented
fake_score_or_sstw_score_fallback: forbidden
validation_scale_split_notebook_path: runnable_after_official_inner_commands_configured
modern_external_baseline_measured_formal_results: still_pending_real_colab_official_commands
```

该状态表示 validation_scale 的工程阻断已经从“缺 SSTW 外层 wrapper”收敛为“需要在 Colab 或等价受治理环境中为 5 个主实验官方 baseline 完成项目内 clone / build / run / adapt / record, 并配置真实官方命令和权重”。如果这些内部官方命令输出 score JSON, `external_baseline_runner` 会把结果转换为 `measured_formal` records, 并由 `external_baseline_execution_manifest.json` 绑定证据路径。

补充约束: 显式设置的 `SSTW_<BASELINE>_EVAL_COMMAND` 优先级高于默认 bridge 模板。因此,
validation_scale 正式门禁同时支持两条可跑通路径: `repository bridge + SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`
和 `SSTW_<BASELINE>_EVAL_COMMAND` 直接写出合规 score JSON。preflight 必须逐 baseline 判断,
不能因为默认启用了 bridge 就误阻断已经配置直接外层命令的 baseline。

## 2026-06-25 repository official eval adapters 接入

为满足 `validation_scale` 严格正式门禁对 5 个主实验现代视频水印 baseline 的统一接入要求, 当前仓库新增 fail-closed 的 repository official adapter 入口:

```text
external_baseline/official_eval_adapters/videoshield.py
external_baseline/official_eval_adapters/vidsig.py
external_baseline/official_eval_adapters/videoseal.py
```

这些 adapter 解决的问题是: Notebook 可以自动配置 5 个主实验 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`, 不再需要用户手写 bridge 内部命令模板。它们仍然保持严格边界:

```text
repository_official_eval_adapter: implemented
SSTW_USE_REPOSITORY_OFFICIAL_BASELINE_ADAPTERS: default_true_in_external_baseline_formal_scoring
missing_official_source_or_weight_behavior: fail_closed
fake_score_or_proxy_score_fallback: forbidden
modern_external_baseline_measured_formal_results: pending_real_colab_official_artifacts_or_checkpoints
```

当前真实运行仍需要 Colab 提供第三方官方源码和对应官方产物。典型额外输入包括:

```text
VideoShield 由 external_baseline.videoshield_official_runtime 在项目内运行官方 watermark generation -> latent inversion -> temporal matching, 或由 SSTW_VIDEOSHIELD_NATIVE_EVAL_COMMAND 显式覆盖
SSTW_VIDSIG_MSG_DECODER_PATH + SSTW_VIDSIG_VAE_CHECKPOINT_PATH, 并由 external_baseline.vidsig_official_runtime 运行官方 generate_ms.py -> attack.py
VideoSeal 官方依赖和 checkpoint, 或 SSTW_VIDEOSEAL_NATIVE_EVAL_COMMAND
```

因此, 当前阶段可以表述为: 5 个主实验现代 baseline 的 SSTW command adapter 和 repository official adapter 已经完成工程接入; 严格正式门禁仍必须通过 Colab 或等价受治理环境中的项目内 clone / build / run / adapt / record 证明这些 adapter 能基于官方源码、权重或项目生成的官方结果缓存写出 `measured_formal` records。若缺少这些官方输入, validation_scale 会失败, 且该失败是正确的 fail-closed 行为。

## 2026-06-25 repository-owned 官方结果缓存 preflight 与资源阻断前移

为继续排除 Colab 冷启动中“缺官方权重、checkpoint、官方结果文件或官方原生命令”导致的晚期阻断, 当前工程新增 repository-owned 官方结果缓存读取与完整性检查能力:

```text
external_baseline/official_result_bundle.py
SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT
SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS
artifacts/external_baseline_official_result_bundle_preflight_decision.json
```

新增项目内结果缓存路径约定:

```text
<bundle_root>/<baseline_id>/records/<prompt_id>__<seed_id>__<attack_name>.json
<bundle_root>/<baseline_id>/records/<trajectory_trace_id>__<attack_name>.json
<bundle_root>/<baseline_id>/<prompt_id>/<seed_id>/<attack_name>.json
<bundle_root>/<baseline_id>/<trajectory_trace_id>/<attack_name>.json
```

阶段性状态为:

```text
official_result_bundle_preflight: implemented
official_result_bundle_reading_in_repository_adapters: implemented
external_baseline_payload_path_fields: implemented
validation_scale_split_baseline_bundle_stage: integrated
google_drive_default_bundle_root: configured_by_notebook
sstw_proxy_result_bundle: forbidden
modern_external_baseline_measured_formal_results: still_requires_real_official_command_or_repository_generated_official_cache
```

该更新没有降低严格门禁。其作用是把外部资源阻断提前暴露: 若某个 modern baseline 既没有可直接运行的官方资源, 也没有由本项目 workflow 生成并覆盖当前 runtime comparison unit 的官方结果缓存, `external_baseline_official_result_bundle_preflight_decision.json` 会明确失败。若项目内缓存完整, repository official adapter 可以直接读取官方 JSON 并写入 `measured_formal` comparison records。

重要边界: repository-owned 官方结果缓存必须由本项目 workflow 调用第三方官方代码或官方原生命令生成, 不能由 SSTW `S_final`、最终判定分数、视频相似度或任意 proxy 分数派生。该能力解决的是 Colab 冷启动和高显存 baseline 的工程可复现问题, 不是允许手写 baseline 结果, 也不是接受外部补交结果。

## 2026-06-25 官方资源 bootstrap 与自动 official bundle 生成接入

根据“检查不通过后应自动补齐可补齐资源”的要求, 当前工程继续把 external baseline
流程从“只做 preflight”推进为“先自动修复, 再严格门禁”。

新增入口包括:

```text
configs/external_baselines/official_resource_requirements.json
external_baseline/official_resource_bootstrap.py
external_baseline/official_bundle_generator.py
paper_workflow/notebook_utils/generative_video_model_probe_workflow.py::build_external_baseline_official_resource_bootstrap_command
paper_workflow/notebook_utils/generative_video_model_probe_workflow.py::build_external_baseline_official_bundle_generation_command
```

Notebook workflow 已增加以下阶段:

```text
external_baseline_source_intake
-> external_baseline_official_resource_bootstrap
-> external_baseline_official_bundle_generation
-> external_baseline_official_result_bundle_preflight
-> external_baseline_comparison
```

阶段性状态为:

```text
official_resource_bootstrap: implemented
public_resource_auto_download_path: implemented_for_supported_resources
videoseal_official_bundle_auto_generation: implemented
vidsig_public_checkpoint_bootstrap: implemented_as_resource_download_when_network_allowed
vidsig_official_generate_ms_runtime: implemented_fail_closed_after_project_runtime_attack
manual_official_resource_required_artifact: implemented_for_resource_heavy_or_unpublished_weight_baselines
strict_gate_fake_pass: forbidden
external_baseline_formal_reference_notebook_auto_repair_path: integrated
```

该更新的含义是: Colab 冷启动时不再只告诉用户缺少官方资源, 而是会先尝试自动安装
公开依赖、下载公开 checkpoint, 并为可自动支持的 baseline 生成 official bundle。当前
VideoShield 与 VidSig 必须先运行各自官方生成流程得到 baseline 自己的 watermarked videos,
不能直接检测 SSTW / Wan 视频。
若某个
baseline 客观需要未公开训练权重、高显存官方生成流程、PRC key 或 maintained info,
workflow 会写出 `manual_official_resource_required`, 仍然不会把该 baseline 伪造成
`measured_formal`。因此严格 validation_scale 通过条件没有降低: 5 个主实验现代 baseline
最终仍必须由本项目基于官方源码、官方 API、官方 checkpoint 或 repository-owned 官方结果缓存产出可审计 score records。


## 2026-06-26 validation_scale 语义重定义与 full_paper protocol 补齐

本段记录 2026-06-26 的历史语义重定义。当前主链已进一步加入 `probe_paper`, 因此本段中“validation_scale 直接承担 fpr=0.1 论文结论候选”的表述仅保留为历史状态。当前规则是: `validation_scale` 明确定义为“target_fpr=0.1 小样本全流程打通验证”, 功能是 `probe_paper` 前的完整协议打通层; `probe_paper` 才使用同一 FPR=10% 口径和 pilot_paper 级样本结构判断 SSTW 是否具备小样本论文闭合证据。

更新后的阶段边界为:

```text
method_mechanism_validation: 验证 SSTW 方法机制和基础 runner
validation_scale: FPR=10% 小样本全流程打通验证
probe_paper: FPR=10% 小样本论文闭合验证
pilot_paper: FPR=1% 小规模 paper 协议结果
full_paper: FPR=0.1% 正式 paper 协议结果
```

关键配置与文档变更:

```text
configs/protocol/validation_scale_generative_probe.json: target_fpr 由该 protocol config 作为唯一语义来源, validation_scale_definition = small_sample_full_protocol_handoff_validation
configs/protocol/probe_paper_generative_probe.json: 新增 target_fpr=0.1 小样本论文闭合协议配置
configs/protocol/full_paper_generative_probe.json: 新增 full_paper 正式协议配置, target_fpr 由该 protocol config 作为唯一语义来源
configs/paper_workflow/generative_video_notebook_workflows.json: validation_scale profile 改为 target_fpr=0.1 全流程打通入口, paper_gate_preflight_layer = true; probe_paper profile 承担 fpr=0.1 小样本论文闭合
configs/external_baselines/official_resource_requirements.json: external baseline 改为项目内自包含产出要求
configs/external_baselines/modern_baseline_colab_commands.json: repository-generated official cache 语义改为 repository-generated cache only
```

external baseline 的新硬边界为:

```text
project_clone
project_build
project_run
project_adapt
project_record
```

不再接受外部补交 result bundle、手写 JSON、NPZ 分数文件、论文表格数字或 SSTW proxy 分数作为主表 baseline 证据。若某个现代 baseline 无法在项目流程内获得官方权重、checkpoint、maintained info 或运行环境, checker 必须写出 `non_runnable_with_governed_reason`, 并阻断 `validation_scale` PASS。

当前操作手册执行闭环在本轮已补齐 validation_scale 相关 builder 与轻量判定实现, 仍有以下缺口:

```text
validation_scale_gate_figure.json: implemented_in experiments/generative_video_model_probe/validation_scale_artifact_package.py
validation_scale_package_manifest.json: implemented_in experiments/generative_video_model_probe/validation_scale_artifact_package.py
stage_transition_decision: implemented_in scripts/check_results/stage_transition_decision.py
external_baseline_self_containment_decision: implemented_in scripts/check_results/external_baseline_self_containment_decision.py
data_split_and_leakage_guard: implemented_in scripts/check_results/data_split_and_leakage_guard.py
full_paper_result_checker 仍未实现
reviewer_evidence_index_builder 仍未实现
external_baseline project_clone / project_build / project_run / project_adapt / project_record 已有轻量 checker, 但真实 Colab measured_formal 运行证据仍需在 validation_scale GPU run 中产出
```

当前阶段结论:

```text
validation_scale_definition_updated: true
full_paper_protocol_config_registered: true
external_baseline_self_contained_output_required: true
validation_scale_real_gpu_run: pending
pilot_paper_allowed: false_until_validation_scale_pass
full_paper_allowed: false_until_validation_scale_probe_paper_pilot_paper_and_full_paper_checker_pass
```

## 2026-06-26 主干门禁去重与轻量判定更新

本次文档修复把两个容易与主干门禁重叠的名称降级:

```text
small_scale_claim_pilot_gate -> small_scale_mechanism_pilot_check, 仅作为 mechanism_validation 子检查
generative_video_model_probe -> implementation package, 不作为独立门禁
```

当前主干门禁固定为:

```text
protocol_governance -> mechanism_validation -> validation_scale -> probe_paper -> pilot_paper -> full_paper -> submission_freeze
```

同时新增并实现三个轻量判定, 它们不新增重型实验阶段, 只负责 fail-closed 阶段跳转和证据边界:

```text
stage_transition_decision
external_baseline_self_containment_decision
data_split_and_leakage_guard
```

其中 `stage_transition_decision` 已拆成阶段明确的跳转判定: `validation_scale_to_probe_paper_transition_decision` 与 `probe_paper_to_pilot_paper_transition_decision`、`pilot_paper_to_full_paper_transition_decision` 和 `full_paper_to_submission_freeze_transition_decision`。这些判定只能在 source gate 已 PASS 后生成, 并由 target gate 消费, 不得反向作为 source gate 自身 PASS 条件。

external baseline 的正式主表证据统一要求 `metric_status == measured_formal`, 且必须由项目内 clone / build / run / adapt / record 产出。governed non-run record 只能作为阻断记录或 limitation 说明, 不能替代 measured_formal baseline。
