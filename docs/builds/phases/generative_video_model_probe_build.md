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

该历史状态表示工程层面已经从真实视频生成推进到攻击、检测、gate 和 package 的完整闭环, 但当时正式实验结论仍等待 `motion_threshold_calibration` 完成。当前最新 motion calibration 与 pilot profile 均已 PASS, 因此本阶段下一步不再是重算 pilot gate, 而是进入 validation-scale 规划与工程门禁构建。

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

该变更属于输入设计修复, 不改变 detector 或 gate 阈值, 也不允许回头修改旧 run 记录。后续 pilot 复跑已经用新 generation 和 formal metric 解除 `formal_motion_claim_ready` 与 `seed_coverage_ready` 缺口; 旧 run 仍保留为历史失败记录。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | pilot 已通过, validation-scale 未完成 |
| 主要差距项 | validation-scale 尚未运行; 现代外部 baseline 目前只有 governed 状态记录, 尚无可进主表的 runnable 结果。 |
| 下一步构建方向 | 构建 validation-scale 真实模型实验、现代 baseline adapter contract、内部消融和 fixed-FPR CI reporter。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: pilot 已通过, validation-scale 未完成
gap_item: validation-scale 尚未运行; 现代外部 baseline 目前只有 governed 状态记录, 尚无可进主表的 runnable 结果。
next_action: 构建 validation-scale 真实模型实验、现代 baseline adapter contract、内部消融和 fixed-FPR CI reporter。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```


## 2026-06-23 最新原生复跑状态

`generative_video_model_probe_colab` 已在 Wan2.1 L4 环境完成 small-scale pilot 原生复跑。最新 package 为:

```text
generative_video_model_probe_colab_20260623_134119_839da169.zip
```

关键状态:

```text
implementation_decision: PASS
effective_mechanism_decision: PASS
mechanism_decision_source: small_scale_claim_pilot_gate
small_scale_pilot_gate_decision: PASS
runtime_attack_decision: PASS
runtime_detection_decision: PASS
record_protocol_missing_failures: []
```

当前阶段下一步不再是修复 pilot, 而是执行 validation-scale 设计和工程化门禁:

```text
validation-scale prompt / seed / attack manifest
modern external baseline governed status and adapter contract
internal ablation matrix
fixed-FPR threshold protocol
statistical confidence interval reporter
```

## 2026-06-23 validation-scale gate 工程入口

当前仓库已新增 validation-scale gate 自动审计入口:

```text
experiments/generative_video_model_probe/validation_scale_gate.py
configs/protocol/validation_scale_generative_probe.json
records/validation_scale_gate_records.jsonl
tables/validation_scale_gate_table.csv
artifacts/validation_scale_gate_decision.json
reports/validation_scale_gate_report.md
```

该 gate 的作用是检查 pilot 通过后是否已经具备进入 full_paper dry-run checker 的条件。它只读取已经落盘的 governed records 和 decision artifacts, 不运行 GPU, 也不补造缺失的 baseline、消融、adaptive attack、replay/sketch 或 CI 结果。

当前 validation-scale gate 检查的最小条件为:

```text
small_scale_claim_pilot_gate_passed
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

同时, Colab 入口已新增 `PROFILE = validation_scale` 运行配置:

```text
prompt_count: 8
seed_per_prompt: 3
expected_generation_count: 24
profile_name: validation_scale
```

需要强调的是, 即使 `validation_scale_gate_decision = PASS`, 也只表示可以进入 `full_paper_dry_run_checker`; 它仍不允许直接生成 full_paper 论文主表。

## 2026-06-23 validation-scale 后处理工程闭环

为延后真实 GPU 复跑但继续推进工程闭环, 当前仓库已补齐三个不依赖重新生成视频的 validation-scale 后处理入口:

```text
experiments/generative_video_model_probe/validation_internal_ablation.py
experiments/generative_video_model_probe/statistical_confidence_interval.py
experiments/generative_video_model_probe/validation_artifact_rebuild.py
```

这些入口分别写出:

```text
records/validation_internal_ablation_records.jsonl
tables/validation_internal_ablation_table.csv
artifacts/validation_internal_ablation_decision.json
reports/validation_internal_ablation_report.md

records/statistical_confidence_interval_records.jsonl
tables/statistical_confidence_interval_table.csv
artifacts/statistical_confidence_interval_decision.json
reports/statistical_confidence_interval_report.md

records/validation_artifact_rebuild_dry_run_records.jsonl
tables/validation_artifact_rebuild_dry_run_table.csv
artifacts/validation_artifact_rebuild_dry_run_decision.json
reports/validation_artifact_rebuild_dry_run_report.md
```

该实现的边界是:

```text
validation_internal_ablation: 使用 runtime detection proxy 生成 validation 级消融矩阵, 不替代 full-paper 正式消融。
statistical_confidence_interval: 只计算 validation runtime detection proxy 的 Wilson 区间, 不替代 FPR=0.001 大规模统计。
validation_artifact_rebuild: 只检查 validation 产物是否具备 records -> tables / reports 重建闭环, 不生成 full-paper package。
```

Colab workflow 已按以下顺序接入:

```text
runtime detection
validation internal ablation
statistical confidence interval
validation artifact rebuild dry-run
validation-scale gate
package to Google Drive
```

因此, 后续真实复跑可以延后; 在复跑发生后, 新增后处理会自动把 validation-scale 缺口显式落盘, 而不是让缺口停留在人工判断。


## 2026-06-23 external_baseline adapter comparison 接入

Colab workflow 已在 runtime detection 之后新增 external baseline comparison 步骤:

```text
runtime detection
external_baseline adapter comparison
small-scale claim pilot gate
validation internal ablation
statistical confidence interval
validation artifact rebuild dry-run
validation-scale gate
package to Google Drive
```

该步骤调用:

```text
python -m experiments.generative_video_model_probe.external_baseline_runner --run-root <drive_run_root> --mode comparison
```

产物包括:

```text
records/external_baseline_score_records.jsonl
tables/external_baseline_comparison_table.csv
artifacts/external_baseline_comparison_decision.json
reports/external_baseline_comparison_report.md
```

该实现的职责是闭合外部 baseline 对比工程链路。当前 measured adapter 仅包括显式 DTW 与 frame matching 同步 control proxy; 现代视频水印 baseline 仍需后续接入官方 adapter。
