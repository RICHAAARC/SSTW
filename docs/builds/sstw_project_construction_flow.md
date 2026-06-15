# SSTW 项目构建流程：面向生成式视频轨迹的密钥条件状态空间水印推断

## 0. 文档定位

本文档用于指导 SSTW 从零构建为一篇可独立投稿 CVPR / ICCV / ECCV 等顶级会议的生成式视频水印论文。本文档的核心目标不是复现或扩展任何已有真实视频同步水印工程, 而是构建一个独立问题、独立方法、独立实验主线的研究项目。

SSTW 的正式研究对象定义为:

\[
\boxed{
\text{Key-conditioned state-space watermark inference over generative video trajectories}
}
\]

中文表述为:

> 面向生成式视频轨迹的密钥条件状态空间水印推断。

项目最终应证明: 视频生成水印中的时间失同步不是简单的 offset / scale 对齐问题, 而是攻击扰动、生成轨迹、密钥编码和水印证据共同作用下的隐状态估计问题。

---

## 1. 顶会投稿版本的最低定义

SSTW 只有在满足以下条件后, 才进入 CVPR / ICCV / ECCV 主会投稿路径。

### 1.1 必须成立的主贡献

```text
key_conditioned_tubelet_code
key_conditioned_state_space_inference
generative_trajectory_observation
key_state_evidence_admissibility
fixed_low_fpr_detector
```

其中, `generative_trajectory_observation` 必须是主贡献之一, 不能只是附加增强模块。

### 1.2 必须完成的主实验

```text
generative_video_model_probe
trajectory_observation_ablation
state_space_model_ablation
key_condition_ablation
fixed_low_fpr_audit
cross_prompt_seed_motion_generalization
quality_motion_semantic_consistency
```

真实视频 VAE latent 和 synthetic latent 只能作为机制 sanity check 或附录证据, 不能作为 SSTW 的主实验支撑。

### 1.3 强接收版本建议

若目标是强接收级别, 建议额外完成:

```text
cross_generation_model_validation
sampling_time_weak_constraint
unseen_attack_type_generalization
unseen_key_generalization
trajectory_control_experiments
failure_case_taxonomy
```

其中 `sampling_time_weak_constraint` 可以作为上限模块, 但只有在质量、运动一致性和语义一致性均不崩溃时才能写入主贡献。

---

## 2. 与其他视频同步水印工作的非重叠边界

SSTW 必须遵守以下边界, 以降低实质性重叠风险。

### 2.1 不能作为主贡献的内容

以下内容不能单独支撑 SSTW 的主贡献:

```text
real_video_vae_latent_probe
explicit_temporal_alignment
explicit_alignment_payload_safety
frame_prc_vs_tubelet_only_vs_explicit_sync_family
h264_h265_real_video_robustness_only
external_video_watermark_baseline_only
```

这些内容可以作为受控 baseline、sanity check 或补充实验, 但不得成为主论文的中心叙事。

### 2.2 必须成为中心的内容

SSTW 的主表、主图和摘要结论必须围绕以下内容展开:

```text
state_space_hidden_state_estimation
generative_trajectory_observation
key_conditioned_state_transition
posterior_state_smoothing
key_state_evidence_admissibility
trajectory_control_suppression
cross_prompt_seed_model_generalization
```

如果主表仍然主要展示真实视频 VAE latent 下的显式同步收益, 则 SSTW 不应按顶会主论文投稿。

### 2.3 并行投稿安全规则

如果另一篇相关真实视频同步水印论文仍处于审稿中, SSTW 的投稿包必须满足:

1. 方法章节不描述另一篇工作的完整机制。

2. 主表不复用另一篇工作的主表结构。

3. 主图不复用另一篇工作的同步机制图。

4. 主实验不以真实视频 VAE latent 对齐攻击为核心。

5. 若会议要求披露 concurrent submission, 需按会议匿名政策在补充材料中说明差异。

---

## 3. 总体构建里程碑

建议将 SSTW 项目拆分为以下语义里程碑。每个里程碑必须产出 records、thresholds、tables、reports 和 mechanism decision。所有正式表格必须从 records / thresholds 重建, 不允许手工拼表。

```text
protocol_freeze
synthetic_state_inference_sanity
real_video_latent_transfer_check
state_space_inference_formalization
trajectory_observation_core_probe
generative_video_model_probe
sampling_time_constraint_probe
submission_package_freeze
```

---

## 4. protocol_freeze

### 4.1 目标

冻结 SSTW 的独立协议框架, 包括 sample role、split、record schema、state trace schema、trajectory trace schema、threshold protocol、attack matrix 和 output layout。

本里程碑只验证协议是否可运行、可审计、可复现, 不验证方法性能。

### 4.2 必须实现

```text
configs/protocol/state_space_watermark_protocol.json
configs/protocol/fixed_low_fpr.json
configs/records/state_space_event_record_schema.json
configs/records/state_trace_schema.json
configs/records/trajectory_trace_schema.json
main/protocol/calibrator.py
main/protocol/record_writer.py
main/protocol/table_builder.py
experiments/state_space_watermark_protocol_probe/runner.py
```

### 4.3 必须固定的 sample role

```text
clean_negative
attacked_negative
watermarked_positive
attacked_positive
```

### 4.4 必须固定的 split

```text
dev
calibration
test
```

### 4.5 必须记录的核心字段

```text
sample_id
split
sample_role
method_variant
attack_name
attack_strength
key_id
content_id
prompt_id
seed_id
generation_model_id
S_payload_raw
S_payload_state
S_state_posterior
S_trajectory_observation
S_final
state_entropy
state_coverage_ratio
state_matched_count
state_posterior_confidence
state_transition_residual
trajectory_consistency_score
key_state_admissibility_status
negative_state_over_threshold_count
threshold_source_split
decision
```

### 4.6 通过标准

```text
threshold 只由 calibration negative 得到
test split 不写 threshold, 不更新融合权重, 不更新 gate 参数
每条 record 包含 split、sample_role、method_variant、attack_name 和 evidence scores
删除 tables 后可由 records 和 thresholds 重建
所有 gate 参数均写入 config 和 manifest
```

### 4.7 产物

```text
records/event_scores.jsonl
thresholds/thresholds.json
artifacts/run_manifest.json
artifacts/runtime_config.json
tables/protocol_smoke_table.csv
reports/protocol_skeleton_report.md
```

---

## 5. synthetic_state_inference_sanity

### 5.1 目标

在 synthetic video latent 上验证 key-conditioned tubelet code 和 state-space inference 是否具有最基本机制有效性。

该里程碑只能回答一个问题:

\[
\text{密钥条件状态估计是否能在受控时间扰动下优于普通时序聚合器?}
\]

### 5.2 实验对象

\[
z\sim\mathcal{N}(0,I),\quad z\in\mathbb{R}^{F\times C\times H\times W}。
\]

### 5.3 必须实现

```text
main/backends/synthetic_video_latent.py
main/methods/state_space_watermark/tubelet_code.py
main/methods/state_space_watermark/state_observation.py
main/methods/state_space_watermark/state_synchronizer.py
main/methods/state_space_watermark/key_state_admissibility.py
main/attacks/synthetic_temporal_attacks.py
experiments/synthetic_state_inference_sanity/runner.py
experiments/synthetic_state_inference_sanity/mechanism_audit.py
```

### 5.4 必须比较的内部 baseline

```text
frame_prc
tubelet_only
explicit_temporal_alignment
generic_temporal_mean_pooling
conv1d_temporal_aggregator
gru_temporal_aggregator
transformer_temporal_aggregator
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
```

### 5.5 必须攻击

```text
no_attack
temporal_crop
local_clip
regular_frame_dropping
irregular_frame_dropping
frame_duplication
speed_change
frame_rate_resampling
segment_jump
latent_gaussian_noise
```

### 5.6 必须证明

1. `tubelet_only` 优于 `frame_prc`, 用于确认 tubelet 结构有基本价值。

2. `key_conditioned_state_space_inference` 优于普通时序聚合器。

3. `key_conditioned_state_space_inference` 优于 `generic_state_space_model` 和 `key_agnostic_state_space_model`。

4. 状态搜索不会导致 attacked negative FPR 失控。

5. state entropy 与失败样本存在可解释关系。

### 5.7 通过标准

```text
synthetic_state_inference_implementation_decision = PASS
synthetic_state_inference_mechanism_decision = PASS
key_conditioned_state_space_inference 在至少两类复杂时间攻击上优于 key_agnostic_state_space_model
attacked_negative_FPR <= target_fpr_tolerance
negative_state_over_threshold_count == 0
```

### 5.8 论文使用边界

本里程碑只能作为机制 sanity check 或附录实验, 不能作为顶会主贡献。

---

## 6. real_video_latent_transfer_check

### 6.1 目标

检查 synthetic 中成立的状态估计机制在真实视频 VAE latent 中是否仍然不崩溃。

该里程碑的定位是 transfer check, 不是 SSTW 主实验。

### 6.2 实验流程

```text
source video
-> VAE encode
-> key-conditioned tubelet watermark embedding
-> VAE decode
-> video attack
-> VAE re-encode
-> state-space inference
-> fixed-FPR decision
```

### 6.3 必须实现

```text
main/backends/real_video_vae_latent.py
main/vae/vae_backend.py
main/video/video_io.py
main/attacks/real_video_temporal_attacks.py
main/attacks/compression.py
main/attacks/spatial.py
main/analysis/quality_metrics.py
main/analysis/temporal_metrics.py
experiments/real_video_latent_transfer_check/runner.py
experiments/real_video_latent_transfer_check/artifact_builder.py
experiments/real_video_latent_transfer_check/mechanism_audit.py
```

### 6.4 必须比较

```text
frame_prc
tubelet_only
explicit_temporal_alignment
key_agnostic_state_space_model
generic_state_space_model
key_conditioned_state_space_inference
```

### 6.5 必须攻击

```text
no_attack
h264_compression
h265_compression
spatial_resize
crop_resize
gaussian_noise
blur
temporal_crop
local_clip
regular_frame_dropping
irregular_frame_dropping
frame_duplication
speed_change
frame_rate_resampling
vae_reconstruction
```

### 6.6 通过标准

```text
real_video_latent_transfer_implementation_decision = PASS
real_video_latent_transfer_mechanism_decision = PASS
negative_state_over_threshold_count == 0
key_state_admissibility_status = PASS
quality_not_collapsed = PASS
temporal_consistency_not_collapsed = PASS
```

### 6.7 论文使用边界

该里程碑只用于证明方法不会在真实视频 latent 中失效。若没有后续 trajectory / generation 实验, 不得以该里程碑作为顶会主论文的核心结果。

---

## 7. state_space_inference_formalization

### 7.1 目标

把状态空间推断从工程模块提升为可审计算法, 明确 transition、observation、key condition、filtering / smoothing、admissibility 和 fixed-FPR detector 的关系。

### 7.2 必须实现

```text
main/methods/state_space_watermark/state_transition.py
main/methods/state_space_watermark/state_observation_model.py
main/methods/state_space_watermark/key_conditioner.py
main/methods/state_space_watermark/state_filter.py
main/methods/state_space_watermark/state_smoother.py
main/methods/state_space_watermark/key_state_admissibility.py
experiments/state_space_inference_formalization/runner.py
experiments/state_space_inference_formalization/ablation_builder.py
```

### 7.3 必须比较

```text
no_state_inference
generic_temporal_mean_pooling
conv1d_temporal_aggregator
gru_temporal_aggregator
transformer_temporal_aggregator
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
key_conditioned_state_space_without_admissibility
key_conditioned_state_space_without_key_condition
```

### 7.4 必须证明

1. 状态空间推断不是普通 temporal aggregator。

2. key condition 是必要的。

3. key-state admissibility 是必要的。

4. phase state、evidence state、confidence state、disturbance state 分别有贡献。

5. unseen key、unseen attack strength、unseen attack type 泛化成立。

### 7.5 通过标准

```text
state_space_inference_formal_decision = PASS
key_condition_ablation_gain > 0
admissibility_negative_tail_status = PASS
state_variable_ablation_all_nontrivial = PASS
unseen_key_generalization_status = PASS
```

---

## 8. trajectory_observation_core_probe

### 8.1 目标

验证生成轨迹是否能作为状态空间推断的独立观测项, 而不是后验分数堆叠。

该里程碑是 SSTW 进入顶会投稿路径的最低门槛。

### 8.2 必须实现

```text
main/trajectory/trajectory_observation.py
main/trajectory/trajectory_reconstruction.py
main/trajectory/trajectory_statistic.py
main/trajectory/trajectory_controls.py
main/methods/state_space_watermark/trajectory_state_observation.py
experiments/trajectory_observation_core_probe/runner.py
experiments/trajectory_observation_core_probe/mechanism_audit.py
```

### 8.3 必须比较

```text
key_conditioned_state_space_inference
key_conditioned_state_space_with_trajectory
trajectory_only
trajectory_observation_without_key_condition
generic_state_space_with_trajectory
explicit_temporal_alignment_with_trajectory_fusion
```

### 8.4 必须证明

1. trajectory response 在 \(H_0\) 与 \(H_1\) 之间存在统计分离。

2. trajectory response 与 static tubelet evidence 不高度冗余。

3. `key_conditioned_state_space_with_trajectory` 在 fixed-FPR 下优于不含 trajectory 的状态空间推断。

4. trajectory controls 不能复现主增益。

5. runtime overhead 可接受。

### 8.5 通过标准

```text
trajectory_observation_implementation_decision = PASS
trajectory_observation_mechanism_decision = PASS
trajectory_gain_over_state_space > 0
trajectory_negative_leakage_delta <= 0
abs(correlation(S_trajectory_observation, S_payload_state)) < correlation_threshold
control_suppression_status = PASS
```

### 8.6 顶会 gate

若该里程碑未通过, SSTW 不应以 CVPR / ICCV / ECCV 主会论文投稿。

---

## 9. generative_video_model_probe

### 9.1 目标

把 SSTW 从 latent transfer check 推进到真实 DiT / Flow Matching 视频生成模型, 证明方法确实解决生成式视频水印轨迹中的隐状态推断问题。

该里程碑应成为 SSTW 的主实验核心。

### 9.2 推荐模型路线

```text
first_runnable_model: CogVideoX 或同类公开视频 DiT 模型
main_generation_model: 具有明确 DiT 或 Flow Matching 属性的视频生成模型
cross_model_validation: Open-Sora / HunyuanVideo / 其他可运行开放模型
```

具体模型选择应以可复现、可记录模型版本、可导出中间轨迹或可重建近似轨迹为前提。

### 9.3 必须比较

```text
no_watermark_or_clean_negative
frame_prc
tubelet_only
explicit_temporal_alignment
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
key_conditioned_state_space_with_trajectory
keyed_state_trajectory_constraint
external_video_watermark_baselines
```

注意: `keyed_state_trajectory_constraint` 只能在 sampling-time weak constraint 通过质量审计后进入主表。

### 9.4 外部 baseline

优先考虑:

```text
VideoMark_style_temporal_matching
VideoMark
RivaGAN
VIDSTAMP
VideoShield
SIGMark
classical_temporal_registration
```

若某个 baseline 不可运行或许可证 / 权重不可复现, 只能进入 limitation report, 不得支撑正向优越性 claim。

### 9.5 必须证明

1. 真实生成视频中仍可保持 fixed low-FPR 检测。

2. key-conditioned state-space inference 优于 explicit temporal alignment。

3. trajectory observation 在生成模型上提供独立增益。

4. prompt、seed、motion pattern、video length 泛化稳定。

5. 质量、motion consistency、semantic consistency 可控。

6. 跨生成模型验证至少存在一个可复现实验包。

### 9.6 通过标准

```text
generative_video_model_implementation_decision = PASS
generative_video_model_mechanism_decision = PASS
generation_model_main_table_ready = true
trajectory_observation_gain_confirmed = true
fixed_low_fpr_audit_pass = true
quality_motion_semantic_consistency_pass = true
cross_prompt_seed_generalization_pass = true
```

---

## 10. sampling_time_constraint_probe

### 10.1 目标

在生成采样过程中加入弱水印约束, 验证其是否能增强 trajectory-aware state observation, 同时保持视觉质量、运动一致性和语义一致性。

该里程碑是强接收版本的上限模块, 不是顶会投稿的最低门槛。

### 10.2 必须实现

```text
main/generation/sampling_hook.py
main/generation/lambda_schedule.py
main/generation/velocity_projection_constraint.py
main/analysis/motion_artifact_audit.py
main/analysis/semantic_consistency_audit.py
experiments/sampling_time_constraint_probe/runner.py
```

### 10.3 必须比较

```text
key_conditioned_state_space_with_trajectory
keyed_state_trajectory_constraint
trajectory_constraint_without_admissibility
trajectory_constraint_without_key_condition
trajectory_constraint_with_lambda_schedule_ablation
```

### 10.4 必须证明

1. 提升 \(S_{\mathrm{trajectory\_observation}}\)。

2. 提升 attacked positive TPR。

3. 不破坏 target FPR。

4. 视频质量下降可控。

5. motion artifact 不明显。

6. prompt / semantic consistency 不明显下降。

### 10.5 失败处理

若该模块失败, 应降级为探索性附录或 future work, 不影响 SSTW 以 state-space inference + trajectory observation 为主线投稿。

---

## 11. submission_package_freeze

### 11.1 目标

冻结完整论文协议, 生成投稿所需主表、消融表、攻击曲线、质量表、runtime 表、failure analysis、claim audit 和 release package。

### 11.2 必须产物

```text
records/event_scores.jsonl
thresholds/thresholds.json
artifacts/run_manifest.json
artifacts/state_space_method_manifest.json
artifacts/claim_audit.json
tables/main_fixed_fpr_table.csv
tables/temporal_attack_breakdown_table.csv
tables/state_space_ablation_table.csv
tables/key_condition_ablation_table.csv
tables/trajectory_observation_ablation_table.csv
tables/trajectory_control_table.csv
tables/generation_model_main_table.csv
tables/cross_prompt_seed_generalization_table.csv
tables/cross_model_generalization_table.csv
tables/external_baseline_comparison_table.csv
tables/quality_motion_semantic_table.csv
tables/runtime_efficiency_table.csv
figures/method_overview.pdf
figures/state_inference_diagram.pdf
figures/trajectory_observation_diagram.pdf
figures/temporal_attack_curve.pdf
figures/trajectory_score_distribution.pdf
figures/quality_robustness_tradeoff.pdf
figures/generated_video_case_grid.pdf
reports/protocol_report.md
reports/mechanism_ablation_report.md
reports/failure_case_report.md
reports/claim_audit_report.md
packages/state_space_trajectory_watermarking_release.tar.zst
```

### 11.3 最终通过标准

```text
所有 supported claim 均映射到具体表格、曲线、报告或 manifest
所有阈值只来自 calibration negative
attacked negative 进入 calibration negative
state-space inference 相比 explicit temporal alignment 与 generic temporal aggregator 具有机制优势
trajectory observation 具有独立增益
sampling-time weak constraint 若写成贡献, 必须通过质量、运动和语义一致性审计
删除 tables / figures / reports 后可由 records 重建全部论文产物
```

---

## 12. 最终投稿判断

### 12.1 不建议投稿顶会的情况

```text
只完成 synthetic_state_inference_sanity
只完成 real_video_latent_transfer_check
主结果仍然围绕 explicit_temporal_alignment 的改进
trajectory_observation_core_probe 未通过
generative_video_model_probe 未通过
fixed_low_fpr_audit 未通过
```

### 12.2 可以投稿顶会的最低情况

```text
state_space_inference_formalization = PASS
trajectory_observation_core_probe = PASS
generative_video_model_probe = PASS
fixed_low_fpr_audit = PASS
claim_audit = PASS
```

### 12.3 强接收版本建议

```text
cross_generation_model_validation = PASS
sampling_time_constraint_probe = PASS
trajectory_control_experiments = PASS
unseen_key_attack_prompt_generalization = PASS
failure_case_taxonomy_ready = true
release_package_rebuildable = true
```
