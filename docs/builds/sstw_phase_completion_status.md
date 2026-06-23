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

| 阶段 | 当前完成标注 | 当前项目实际情况 | 未完成 / 差距项 | 下一步构建方向 | full_paper 影响 |
|---|---|---|---|---|---|
| `protocol_governance_foundation` | 已完成当前阶段, 持续维护 | 协议、字段注册、测试分层、harness 审计已可运行。 | 新增 baseline、full_paper 和 replay 字段时仍需同步 registry 与 schema。 | 随后续阶段增量维护字段闭包。 | 是所有结果包的前置条件。 |
| `synthetic_state_inference_sanity` | 部分完成 | synthetic runner、state-space 模块和轻量机制测试已存在。 | 不能支撑真实 Flow Matching 视频主 claim。 | 保持为 state inference sanity 与 regression test。 | 只作为机制合理性证据。 |
| `real_video_latent_transfer_check` | 部分完成 | VAE/视频链路模块和 runner 已存在。 | 真实视频 VAE 大规模低误报验证不足。 | 在 pilot 通过后补齐 real-video transfer validation。 | 影响 endpoint robustness 与视频链路可信度。 |
| `state_space_inference_formalization` | 部分完成 | state variable、transition、observation、admissibility 结构已拆分。 | generic SSM、Mamba-style temporal fusion、key-agnostic 对比仍需 full-scale governed records。 | 强化 formal ablation 与 negative tail audit。 | 影响 Claim-2 的状态后验贡献。 |
| `trajectory_observation_core_probe` | 部分完成 | trajectory observation、velocity projection、correlation audit 入口已存在。 | path evidence 独立增益仍需 pilot / full validation 证明。 | 绑定 endpoint、path、velocity 三证据并跑消融。 | 影响 Claim-2 是否成立。 |
| `flow_model_adapter_preflight` | 已完成前置验证 | Wan2.1 callback、time grid、sampler signature 和 latent displacement proxy 已验证。 | 真实 velocity field 原值未必可访问, 当前主要依赖 proxy。 | 保持 proxy 边界, 如能访问真实 velocity 再升级。 | 满足进入 sampling-time 与 pilot 的接口前置。 |
| `sampling_time_constraint_probe` | 已完成机制前置验证 | recommended profile 显示 keyed alignment gain 与 wrong-key 分离。 | 尚不能替代 attack matrix、negative family、fixed-FPR path gain。 | 作为 small-scale pilot 前置证据。 | 证明可进入 pilot, 不直接支撑 full_paper。 |
| `motion_threshold_calibration` | 已完成 engineering calibration | 已有 `motion_delta_calibrated_v1` 可作 pilot guardrail。 | 不是论文级 `TPR@FPR=0.001` fixed-FPR 证据。 | full_paper 前补齐更大 held-out negative 和 CI。 | 影响 motion claim 样本资格过滤。 |
| `small_scale_claim_pilot_gate` | 已完成 small-scale pilot | 最新 Wan2.1 pilot 原生复跑已达到 16/16 eligible、seed_per_prompt_min=2、runtime attack/detection 48/48 ready、pilot_gate_decision=PASS。 | 只能支撑进入 validation-scale, 不能替代 full_paper 或论文级 fixed-FPR。 | 进入 validation-scale generative probe, 并保留 pilot 作为工作流证据。 | 解除 full experiment 前置阻塞, 但不解除 full_paper 阻塞。 |
| `generative_video_model_probe` | pilot 已通过, validation-scale 未完成 | 生成、attack、detection、postprocess、packager 与协议字段闭包已在 Wan2.1 pilot 中通过。 | validation-scale 样本量、现代外部 baseline runnable 结果、内部消融 full-scale records、论文级 fixed-FPR 尚未完成。 | 构建 validation-scale generative probe、现代 baseline 状态/adapter、内部消融与 CI reporter。 | 影响主表、baseline comparison、ablation table 和真实模型结论。 |
| `replay_and_authenticated_sketch_gate` | 未完成 | digest、manifest、trajectory trace 基础模块存在。 | authenticated sketch、replay uncertainty、wrong prompt replay 未闭合。 | 补齐签名 sketch、replay records 和 checker。 | 影响 Claim-3 强度; 不通过则降级 Claim-3。 |
| `flow_specific_adaptive_attack_gate` | 未完成 | phase 文档已补建, 但 runner、manifest 与 governed records 尚未完成。 | adaptive attacks、endpoint-preserving resampling、path cancellation 未形成 records。 | 补齐 runner 设计、stress protocol、attack manifest 和 checker。 | full_paper 前必须完成或明确降级。 |
| `full_paper_result_package_gate` | 未开始 | 仅有文档规范, 尚未实现 dry-run checker / result checker。 | 必须等 validation-scale、现代外部 baseline、adaptive attack、replay/sketch 与 paper-level fixed-FPR 通过后才能运行。 | 建立 dry-run checker、sample-size manifest 和统计 CI reporter。 | 是论文结果包产出前最后阻断 gate。 |
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

需要重新执行 `generative_video_model_probe_colab.ipynb` 的 `PROFILE = motion_calibration` 流程。旧 package 不会自动获得新字段, 必须重新运行 formal metric 与 calibration。


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

`paper_workflow/colab_utils/generative_video_model_probe_colab.ipynb` 已切换到:

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
runs/generative_video_model_probe_colab/artifacts/motion_threshold_calibration_decision.json
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

是否继续进入 full experiment, 必须由新的 pilot gate 结果决定。


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
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260622_174746_e0f9c79d.zip
G:\我的云端硬盘\SSTW\packages\generative_video_model_probe\generative_video_model_probe_colab_20260622_174746_e0f9c79d_package_manifest.json
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
generative_video_model_probe 已进入 validation-scale 准备阶段
full_paper_result_package_gate 未开始, 且在 validation / adaptive attack / baseline / replay-sketch 前不得启动
```

### 当前阻塞项

```text
primary_blocker: validation_scale_generative_probe_not_completed
secondary_blocker: modern_external_baseline_main_comparison_not_ready
secondary_blocker: internal_ablation_full_scale_records_not_ready
secondary_blocker: replay_and_authenticated_sketch_gate_not_closed
secondary_blocker: flow_specific_adaptive_attack_gate_not_closed
secondary_blocker: paper_level_fpr_0_001_calibration_not_ready
secondary_blocker: full_paper_dry_run_checker_not_implemented
```

### 下一步允许执行

```text
validation-scale prompt / seed / attack manifest 规划
validation-scale generative probe runner / checker 增强
modern external baseline governed status records and adapter contracts
internal ablation matrix records
fixed-FPR and confidence interval reporter
flow-specific adaptive attack runner design
replay/sketch verification runner design
full_paper dry-run checker implementation
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

最新原生复跑已经满足上述标准。因此, 后续工作不应继续围绕“修复 pilot”展开, 而应进入 validation-scale 与论文级证据充分性构建。

## 2026-06-23 文档增强后工程化门禁缺口

### 当前文档层面已增强的内容

```text
full_paper dry-run checker 规范已写入总体流程
统计置信区间与 cluster-by-video interval 要求已写入总体流程
审稿风险对照矩阵已写入总体流程
算法原语到 full_paper package 的记录映射已写入算法原语文档
```

### 当前仍未工程化的门禁

```text
full_paper_dry_run_checker: not_implemented
full_paper_result_checker: not_implemented
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
generative video phase 新增 validation-scale 充分性矩阵
adaptive attack phase 新增攻击者知识层级与 claim 降级规则
small-scale pilot phase 新增 pilot 结果使用边界
submission freeze phase 新增 reviewer evidence index 要求
算法原语文档新增可证伪条件和原语级实验打包要求
```

### 当前状态解释

本次增强只修改构建文档, 不产出 full_paper 结果。最新阶段判断为:

```text
primary_blocker: validation_scale_generative_probe_not_completed
full_paper_allowed: false
submission_freeze_allowed: false
```

### 下一步工程化方向

```text
1. 设计并执行 validation-scale generative probe。
2. 将现代外部 baseline 的 governed status record 扩展为 adapter contract 或明确 non-run protocol gap。
3. 将 full_paper dry-run checker、baseline runner、adaptive attack runner、CI reporter 和 reviewer evidence index 从文档规范转化为 repository 工程实现。
```

## 2026-06-23 文档继续增强: full_paper 工程门禁实现规范

### 本次补强内容

```text
新增 sstw_full_paper_engineering_gate_spec.md
总体流程新增 full_paper 工程门禁实现规范索引
顶会实验充分性清单新增工程化 readiness 评分
```

### 当前状态解释

本次补强进一步降低“文档无法落地为 Codex 工程任务”的风险。当前项目阶段已经从 pilot 阻塞转为 validation-scale 阻塞:

```text
primary_blocker: validation_scale_generative_probe_not_completed
full_paper_allowed: false
submission_freeze_allowed: false
```

### 新增工程化目标

后续若继续推进代码实现, 应优先按 `sstw_full_paper_engineering_gate_spec.md` 实现:

```text
full_paper_dry_run_checker
statistical_confidence_interval_reporter
modern_external_baseline_runner
reviewer_evidence_index_builder
full_paper_result_checker
flow_specific_adaptive_attack_runner
```



## 2026-06-23 最新原生复跑后阶段状态更新

### 当前总体判定

最新 Google Drive 落盘结果显示, `small_scale_claim_pilot_gate` 已经通过。当前项目不再处于 small-scale pilot 阻塞状态, 而是进入 validation-scale 与论文级证据充分性构建阶段。

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
generative_package: generative_video_model_probe_colab_20260623_134119_839da169.zip
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
primary_blocker: validation_scale_generative_probe_not_completed
secondary_blocker: modern_external_baseline_main_comparison_not_ready
secondary_blocker: internal_ablation_full_scale_records_not_ready
secondary_blocker: flow_specific_adaptive_attack_gate_not_closed
secondary_blocker: replay_and_authenticated_sketch_gate_not_closed
secondary_blocker: paper_level_fpr_0_001_calibration_not_ready
secondary_blocker: full_paper_dry_run_checker_not_implemented
```

### 下一步允许执行

```text
validation-scale generative video probe planning
modern external baseline governed status records and adapter contracts
internal ablation matrix records
fixed-FPR and confidence interval reporter
flow-specific adaptive attack runner design
full_paper dry-run checker implementation
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
