# generative_video_model_probe 分阶段构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本阶段构建流程, 再说明当前阶段具体完成情况。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本阶段构建流程

### 1.1 阶段目标

该阶段在真实生成式视频模型上验证 SSTW 的轨迹观测、状态空间推断、外部 baseline 和检测协议是否可运行。该阶段应在 `flow_model_adapter_preflight`、`sampling_time_constraint_probe` 与 `small_scale_claim_pilot_gate` 之后进行。该阶段必须服务于 Wan2.1 主线, 轻量模型只能作为工程预验证或 fallback probe。

### 1.2 输入

```text
configs/protocol/generative_video_model_probe.json
configs/generation/generation_models.json
configs/generation/prompts.json
configs/generation/seeds.json
configs/external_baselines/external_baselines.json
experiments/generative_video_model_probe/
paper_workflow/colab_utils/generative_video_model_probe_colab.ipynb
```

### 1.3 构建任务

1. 构造 prompt suite、seed plan、content manifest 和 generation manifest。
2. 调用生成式视频模型产生视频、latent proxy 或 trajectory proxy。
3. 执行 SSTW full method、内部机制 baseline 和外部 baseline。
4. 执行攻击、检测、formal metrics、generalization 和 postprocess。
5. 生成 package 供 Google Drive 下载与后续 submission freeze 调用。

### 1.4 主线模型

```text
Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

该模型用于支撑 Flow Matching / velocity-field sampler 主线。其他轻量模型只能作为冷启动或工程排错路线。

### 1.5 外部 baseline 分层

```text
VideoShield
SIGMark
SPDMark
VideoMark 或 VidSig
VideoSeal
explicit_dtw_temporal_alignment
frame_matching_temporal_registration
```

其中, VideoShield、SIGMark、SPDMark、VideoMark / VidSig 与 VideoSeal 用于覆盖 2025-2026 年现代视频生成水印和后处理视频水印路线; `explicit_dtw_temporal_alignment` 与 `frame_matching_temporal_registration` 只作为显式同步 control, 不能作为顶刊顶会版本的唯一外部 baseline。

### 1.6 必须比较

```text
sstw_full_method
endpoint_only_control
trajectory_only_score
without_velocity_constraint
without_endpoint_aware_control
without_replay_uncertainty_weighting
explicit_dtw_temporal_alignment
frame_matching_temporal_registration
VideoShield
SIGMark
SPDMark
VideoMark_or_VidSig
VideoSeal
generic_ssm_baseline
```

### 1.7 通过标准

1. 主线记录必须说明 generation model 是 Flow Matching / velocity-field sampler 相关模型。
2. velocity / flow trajectory proxy 必须参与同步证据。
3. SSTW full method 在 `TPR@FPR=0.01` 下优于外部 baseline 与内部机制 baseline。
4. full_paper 前置验证必须确认 `TPR@FPR=0.001` 的大规模 records、threshold 和 held-out negative 协议可运行。
5. 质量、运动和语义指标不显示不可接受退化。

### 1.8 validation-scale 运行要求

在进入 full_paper 前, 本阶段必须先完成 validation-scale 运行。validation-scale 不要求达到 full_paper 样本量, 但必须证明完整流程不会阻断。

必须覆盖:

```text
modern external baseline dry-run 或 runnable records
internal ablation matrix dry-run
runtime attack records
runtime detection records
wrong sampler replay records
negative family records
artifact rebuild dry-run
```

### 1.9 modern baseline 接入状态记录

每个现代外部 baseline 必须写入:

```text
external_baseline_name
external_baseline_source_url
external_baseline_runnable_status
external_baseline_adapter_status
external_baseline_input_compatibility_status
external_baseline_output_record_status
external_baseline_not_run_reason
external_baseline_protocol_gap
external_baseline_result_used_for_claim
```

若 baseline 不能运行, 也必须生成 governed record, 不能静默缺失。

### 1.10 validation-scale 充分性矩阵

pilot PASS 后不得直接进入 full_paper。必须先执行 validation-scale 真实模型实验, 用较小但覆盖完整的矩阵验证全链路:

```text
validation_generation_records_ready
validation_attack_records_ready
validation_detection_records_ready
validation_external_baseline_records_ready
validation_internal_ablation_records_ready
validation_adaptive_attack_records_ready
validation_replay_or_sketch_records_ready
validation_confidence_interval_report_ready
validation_artifact_rebuild_dry_run_ready
```

validation-scale 不要求达到 full_paper 的最终样本量, 但必须覆盖 full_paper 的所有产物类型。若某一类产物在 validation-scale 阻断, 不允许进入 full_paper。

### 1.11 现代 baseline 最小接入组合

本阶段进入 full_paper 前, 至少需要确认以下 baseline 组合已经 runnable 或已有 governed non-run reason:

```text
in_generation_or_diffusion_video_watermark_baseline
post_hoc_neural_video_watermark_baseline
explicit_temporal_alignment_control
endpoint_only_control
generic_state_space_or_temporal_aggregator_control
```

如果现代 in-generation baseline 暂时无法运行, 不能用传统 frame watermark 替代。此时应输出:

```text
modern_external_baseline_blocking_status
external_baseline_not_run_reason
claim_downgrade_recommendation
```

并阻止 full_paper 主对比表生成。

### 1.12 baseline runner 工程规范索引

现代 baseline runner 的实现必须遵守:

```text
docs/builds/sstw_full_paper_engineering_gate_spec.md
```

本阶段只允许把以下结果写入主表候选:

```text
external_baseline_runnable_status == runnable
external_baseline_adapter_status == ready
external_baseline_output_record_status == governed_records_written
external_baseline_result_used_for_claim == true
```

无法运行的 baseline 只能生成 governed non-run record, 不得静默删除。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有 generative video probe 相关模块:

```text
experiments/generative_video_model_probe/generation_runner.py
experiments/generative_video_model_probe/detection_runner.py
experiments/generative_video_model_probe/attack_runner.py
experiments/generative_video_model_probe/external_baseline_runner.py
experiments/generative_video_model_probe/formal_metric_runner.py
experiments/generative_video_model_probe/generalization_runner.py
experiments/generative_video_model_probe/postprocess_runner.py
experiments/generative_video_model_probe/package_outputs.py
scripts/check_results/generative_video_colab_result_checker.py
scripts/package_results/generative_video_drive_packager.py
```

### 2.2 当前阶段补充要求

按照新的整体流程, 本阶段需要继续强化 Wan2.1 主线、外部 baseline、内部机制 baseline 和 Google Drive package 的一致性。数据集构造必须与模型测试运行分离。

### 2.3 motion threshold calibration 前置要求

本阶段可以在工程上继续验证 Wan2.1 生成、trajectory capture、attack matrix、negative family 和 external baseline。但如果结果将用于最终论文 claim 或 submission package, 必须先完成或引用冻结的 `motion_threshold_calibration` artifact。

未完成 calibration 时, formal motion gate 只能解释为:

```text
threshold_id: motion_delta_heuristic_v1
threshold_source_split: heuristic_precalibration
usage: pilot_guardrail
```

当前已有可用于 small-scale pilot 的冻结工程阈值:

```text
threshold_id: motion_delta_calibrated_v1
threshold_source_split: calibration
motion_delta_threshold: 0.010607
usage: frozen_engineering_motion_threshold_for_small_scale_pilot
```

禁止将 evaluation split 或每次生成后的结果反向用于动态调整阈值。正式实验必须使用预先校准并冻结的 threshold artifact。

### 2.4 runtime attack / detection 工程闭环

当前 `generative_video_model_probe` 已补齐真实 runtime 文件级 attack 和 attacked video detection scoring 的工程路径。新增或强化的入口包括:

```text
experiments/generative_video_model_probe/attack_runner.py
experiments/generative_video_model_probe/detection_runner.py
paper_workflow/notebook_utils/generative_video_model_probe_workflow.py
paper_workflow/colab_utils/generative_video_model_probe_colab.ipynb
scripts/package_results/generative_video_drive_packager.py
```

Colab notebook 的冷启动顺序现在包含:

```text
prepare prompt suite
run Wan2.1 generation
run formal quality / motion / semantic metrics
reuse frozen motion threshold calibration artifact for pilot profile
run mechanism postprocess
run pilot matrix postprocess
run runtime video-file attack
run runtime attacked video detection
write small-scale claim pilot gate
run pytest and harness
package to Google Drive
```

历史 pilot 落盘 package manifest 曾包含:

```text
runtime_attack_decision: PASS
runtime_attack_ready_count: 48
runtime_detection_decision: PASS
runtime_detection_ready_count: 48
small_scale_pilot_claim_support_status: blocked_until_motion_threshold_calibration
```

该历史状态表示工程层面已经从真实视频生成推进到攻击、检测、gate 和 package 的完整闭环, 但当时正式实验结论仍等待 `motion_threshold_calibration` 完成。最新 motion calibration 已经 PASS, 因此下一轮 pilot profile 应复用冻结 calibration artifact 并重新计算 pilot gate。

### 2.5 formal motion claim 过滤边界

本阶段现在要求所有进入 motion / trajectory claim 的生成样本先通过 formal visual、motion 和 semantic gate。若某个正向运动样本实际生成结果接近静止或低运动:

```text
formal_motion_consistency_ready: false
formal_metric_blocking_reason: formal_motion_consistency_not_ready
```

则该样本:

```text
保留在 generation_records 与 formal records 中
不物理删除原始视频
不进入 pilot matrix claim evidence
不计入 prompt / seed 覆盖率
不计入 runtime attack / detection claim-ready 统计
```

该过滤由以下公共模块实现:

```text
experiments/generative_video_model_probe/formal_motion_claim_filter.py
```

该模块只读取 motion observability、formal readiness 和样本角色字段, 不读取 `S_final`、`S_final_conservative` 或最终判定分数, 因此不会把最终检测结果反向用于污染过滤。

### 2.6 pilot positive motion prompt 修复

当前 `generative_video_model_probe` 的 prompt suite 已从 calibration prompt 修复扩展到 pilot heldout prompt 修复。失败原因是旧的 `heldout_rotation_scene` 使用:

```text
motion_pattern_id: gentle_rotation
```

在真实 Wan2.1 生成中可能被模型实现为近静止或极弱旋转, 导致 formal motion gate 低运动失败。新 prompt 保留同一 `prompt_id`, 但把运动约束改为:

```text
large foreground object
left-to-right translation across the full frame
rapid full rotation
strong visible displacement in every frame
```

该变更属于输入设计修复, 不改变 detector 或 gate 阈值, 也不允许回头修改旧 run 记录。下一次 pilot 必须重新运行 generation 和 formal metric, 才能判断该修复是否解除 `formal_motion_claim_ready` 与 `seed_coverage_ready` 缺口。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 结构就绪, full experiment 未完成 |
| 主要差距项 | pilot 未 PASS 前不得进入 full experiment; 现代外部 baseline 尚未集成。 |
| 下一步构建方向 | pilot PASS 后扩展 validation-scale 真实模型实验和现代 baseline runner。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 结构就绪, full experiment 未完成
gap_item: pilot 未 PASS 前不得进入 full experiment; 现代外部 baseline 尚未集成。
next_action: pilot PASS 后扩展 validation-scale 真实模型实验和现代 baseline runner。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```
