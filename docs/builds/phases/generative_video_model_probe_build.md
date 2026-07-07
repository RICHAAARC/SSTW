# generative_video_model_probe 实现 package 构建流程

本文档是 SSTW 分阶段构建流程文档。文档结构固定为: 先说明本实现 package 构建流程, 再说明当前具体完成情况。`generative_video_model_probe` 不再作为独立主门禁使用, 只表示真实生成式视频模型实验 package。阶段放行必须服从主干门禁: `protocol_governance -> mechanism_validation -> validation_scale -> pilot_paper -> full_paper -> submission_freeze`。

本阶段文档服从 `docs/builds/sstw_project_construction_flow.md` 与 `docs/builds/sstw_method_mechanism_design.md`。阶段完成情况只描述仓库当前可观察的工程、配置、测试和文档状态, 不把临时实验输出写成论文结论。

## 1. 本实现 package 构建流程

### 1.1 阶段目标

该实现 package 在真实生成式视频模型上验证 SSTW 的轨迹观测、状态空间推断、外部 baseline 和检测协议是否可运行。它应在 `flow_model_adapter_preflight`、`sampling_time_constraint_probe` 与 small-scale 机制 pilot 检查之后使用, 但不再单独放行 paper 阶段。它必须服务于 Wan2.1 主线, 轻量模型只能作为工程预验证或 fallback probe。

### 1.2 输入

```text
configs/protocol/generative_video_model_probe.json
configs/generation/generation_models.json
configs/generation/prompts.json
configs/generation/seeds.json
configs/external_baselines/external_baselines.json
configs/paper_workflow/generative_video_notebook_workflows.json
experiments/generative_video_model_probe/
paper_workflow/colab_notebooks/motion_threshold_calibration_colab.ipynb
paper_workflow/colab_notebooks/generative_video_runtime_colab.ipynb
paper_workflow/colab_notebooks/videoseal_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/vidsig_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/videoshield_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/revmark_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/wam_frame_formal_reference_colab.ipynb
paper_workflow/colab_notebooks/formal_comparison_scoring_colab.ipynb
paper_workflow/colab_notebooks/paper_evidence_postprocess_colab.ipynb
paper_workflow/colab_notebooks/paper_gate_and_package_colab.ipynb
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
VideoSeal
explicit_dtw_temporal_alignment
frame_matching_temporal_registration
```


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
VideoSeal
generic_ssm_baseline
```

### 1.7 通过标准

1. 主线记录必须说明 generation model 是 Flow Matching / velocity-field sampler 相关模型。
2. velocity / flow trajectory proxy 必须参与同步证据。
3. SSTW full method 在 `TPR@FPR=0.01` 下优于外部 baseline 与内部机制 baseline。
4. full_paper 前置验证必须确认 `TPR@FPR=0.001` 的大规模 records、threshold 和 held-out negative 协议可运行。
5. 质量、运动和语义指标不显示不可接受退化。

### 1.8 validation_scale 运行要求

在进入 pilot_paper 和 full_paper 结果运行前, 本实现 package 必须先服务于 `validation_scale` 运行。`validation_scale` 是 target_fpr=0.1 的小样本完整协议论文主张候选门禁。它不要求达到 pilot_paper 或 full_paper 样本量, 但必须完成 paper 相关的全部机制构建, 覆盖与 full_paper 一致的 46 个 runtime attack 和 11 个 non-runtime/adaptive 协议, 并能够在小样本规模上产出全部结果类型。

必须覆盖:

```text
runtime generation records
formal quality / motion / semantic records
runtime attack records
runtime detection records
complete external baseline comparison records
modern external baseline measured_formal records
internal ablation matrix records
flow-specific adaptive attack records
wrong sampler / wrong prompt / wrong time grid replay records
replay/sketch gate records 或受治理 Claim-3 downgrade records
negative family records
fixed-FPR confidence interval report
validation tables / figures / reports
artifact rebuild dry-run
claim audit report
package manifest
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

若 baseline 不能运行, 也必须生成 governed record, 不能静默缺失。但在 `validation_scale` 作为 paper 级前小样本全流程打通门禁时, 必需现代 baseline 只能用 `measured_formal` records 支撑通过; governed non-run record 只能解释阻断原因, 不能替代正式对比结果。

### 1.10 validation_scale 充分性矩阵

历史 small-scale pilot PASS 后不得直接进入 full_paper。必须先执行 `validation_scale` 真实模型实验, 用较小但覆盖完整的矩阵验证全链路:

```text
validation_generation_records_ready
validation_attack_records_ready
validation_detection_records_ready
validation_external_baseline_status_records_ready
validation_external_baseline_comparison_records_ready
validation_modern_external_baseline_formal_records_ready
validation_internal_ablation_records_ready
validation_adaptive_attack_records_ready
validation_replay_or_sketch_records_ready
validation_confidence_interval_report_ready
validation_tables_figures_reports_ready
validation_artifact_rebuild_dry_run_ready
validation_claim_audit_ready
```

`validation_scale` 不要求达到 full_paper 的最终样本量, 但必须覆盖 paper 协议的所有机制和所有产物类型。若某一类产物在 validation_scale 阻断, 不允许进入 `pilot_paper`; 此时只能继续修复缺失机制、adapter 或门禁规则。validation_scale 通过只是进入 pilot_paper 的必要条件, 还必须生成 validation_scale_to_pilot_paper_transition_decision; full_paper 仍需 pilot_paper_gate、pilot_paper_to_full_paper_transition_decision 与 full_paper_result_checker。

### 1.11 现代 baseline 最小接入组合

本实现 package 进入 `pilot_paper` 前, 至少需要确认以下 baseline 组合已经通过项目内 clone / build / run / adapt / record 与正式 adapter 产出同批 comparison records。对必需现代视频水印 baseline 而言, `governed non-run reason` 只能作为阻断解释, 不能作为 `validation_scale` 通过依据:

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

并阻止 `validation_scale` 通过, 同时阻止 `pilot_paper`、`full_paper` 和后续主对比表生成。

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
external_baseline_project_clone_status == completed
external_baseline_project_build_status == completed
external_baseline_project_run_status == completed
external_baseline_project_adapt_status == completed
external_baseline_project_record_status == completed
metric_status == measured_formal
external_baseline_result_used_for_claim == true
```

无法运行的 baseline 只能生成 governed non-run record, 不得静默删除。

## 2. 当前阶段具体完成情况

### 2.1 已有工程文件

当前仓库已有 generative video implementation package 相关模块:

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
paper_workflow/colab_notebooks/generative_video_runtime_colab.ipynb
scripts/package_results/generative_video_drive_packager.py
```

推荐 Colab 冷启动顺序现在拆分为:

```text
motion_threshold_calibration_colab.ipynb
-> generative_video_runtime_colab.ipynb
-> 5 个主实验 modern external baseline formal reference Notebook
-> formal_comparison_scoring_colab.ipynb
-> paper_evidence_postprocess_colab.ipynb
-> paper_gate_and_package_colab.ipynb
```

旧综合 Notebook 已移除; 当前只保留拆分后的 Notebook workflow。
所有 Notebook 的 profile、Drive 目录和 stage plan 均由
`configs/paper_workflow/generative_video_notebook_workflows.json` 控制。

Hunyuan `gen -> extract` 路径: Notebook 调用
与 SSTW runtime records 对齐的 prompt set, 运行官方 `main.py --mode=gen` 与
`main.py --mode=extract`, 再把官方 `*-bit_accuracy.npz` 转写为 project-owned
official bundle。该 bundle 仍需经统一 external baseline runner 转成
`metric_status: measured_formal`, 不能在 Notebook 中手写正式 records。

历史 pilot 落盘 package manifest 曾包含:

```text
runtime_attack_decision: PASS
runtime_attack_ready_count: 48
runtime_detection_decision: PASS
runtime_detection_ready_count: 48
small_scale_pilot_claim_support_status: blocked_until_motion_threshold_calibration
```

该历史状态表示工程层面已经从真实视频生成推进到攻击、检测、gate 和 package 的完整闭环, 但当时正式实验结论仍等待 `motion_threshold_calibration` 完成。当前最新 motion calibration 与 pilot profile 均已 PASS, 因此本阶段下一步不再是重算 pilot gate, 而是进入 validation_scale 规划与工程门禁构建。

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
不进入 small_scale_mechanism_pilot_check claim evidence
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
| 完成状态 | pilot 已通过, validation_scale 未完成 |
| 下一步构建方向 | 构建 validation_scale 真实模型小样本全流程打通实验、现代 baseline adapter contract、内部消融和 fixed-FPR CI reporter。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: pilot 已通过, validation_scale 未完成
gap_item: validation_scale 尚未运行; 现代外部 baseline 目前只有 governed 状态记录, 尚无可进主表的 runnable 结果。
next_action: 构建 validation_scale 真实模型实验、现代 baseline adapter contract、内部消融和 fixed-FPR CI reporter。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```


## 2026-06-23 最新原生复跑状态

`generative_video_runtime` 已在 Wan2.1 L4 环境完成 small-scale pilot 原生复跑。最新 package 为:

```text
generative_video_runtime_20260623_134119_839da169.zip
```

关键状态:

```text
implementation_decision: PASS
effective_mechanism_decision: PASS
mechanism_decision_source: small_scale_mechanism_pilot_check
small_scale_mechanism_pilot_decision: PASS
runtime_attack_decision: PASS
runtime_detection_decision: PASS
record_protocol_missing_failures: []
```

当前阶段下一步不再是修复 pilot, 而是执行 validation_scale 设计和工程化门禁:

```text
validation_scale prompt / seed / attack manifest
modern external baseline governed status and adapter contract
internal ablation matrix
fixed-FPR threshold protocol
statistical confidence interval reporter
```

## 2026-06-23 validation_scale gate 工程入口

当前仓库已新增 validation_scale gate 自动审计入口:

```text
experiments/generative_video_model_probe/validation_scale_gate.py
configs/protocol/validation_scale_generative_probe.json
records/validation_scale_gate_records.jsonl
tables/validation_scale_gate_table.csv
artifacts/validation_scale_gate_decision.json
reports/validation_scale_gate_report.md
```

该 gate 的作用是检查 small_scale_mechanism_pilot_check 通过后是否已经具备进入 paper 级前小样本全流程打通验证的条件。它只读取已经落盘的 governed records 和 decision artifacts, 不运行 GPU, 也不补造缺失的 baseline、消融、adaptive attack、replay/sketch、CI、tables、figures、reports 或 claim audit 结果。

当前 validation_scale gate 检查的最小条件为:

```text
small_scale_mechanism_pilot_check_passed
validation_generation_records_ready
validation_attack_records_ready
validation_detection_records_ready
validation_external_baseline_status_records_ready
validation_external_baseline_comparison_records_ready
validation_modern_external_baseline_formal_records_ready
validation_internal_ablation_records_ready
validation_adaptive_attack_records_ready
validation_replay_or_sketch_records_ready
validation_confidence_interval_report_ready
validation_tables_figures_reports_ready
validation_artifact_rebuild_dry_run_ready
validation_claim_audit_ready
```

同时, Colab 入口已新增 `PROFILE = validation_scale` 运行配置:

```text
prompt_count: 8
seed_per_prompt: 3
expected_generation_count: 24
profile_name: validation_scale
```

需要强调的是, `validation_scale_gate_decision = PASS` 表示 paper 级运行的全部机制已经在小样本规模上闭合, 因而可以生成 `validation_scale_to_pilot_paper_transition_decision` 并进入 `pilot_paper_gate`; 它仍不允许直接生成 full_paper 规模论文主表, 但它必须已经能产出小样本全结果包。

## 2026-06-23 validation_scale 后处理工程闭环

为延后真实 GPU 复跑但继续推进工程闭环, 当前仓库已补齐三个不依赖重新生成视频的 validation_scale 后处理入口:

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
validation_internal_ablation: 使用 runtime detection proxy 生成 validation_scale 或 pilot_paper 同批 trace 的消融矩阵; validation_scale 结果只用于工程稳定性, pilot_paper 结果用于小样本论文级协议预演, 二者都不替代 full_paper 大规模正式消融。
statistical_confidence_interval: 只计算 validation runtime detection proxy 的 Wilson 区间, 不替代 FPR=0.001 大规模统计。
validation_artifact_rebuild: 只检查 validation 产物是否具备 records -> tables / reports 重建闭环, 不生成 full_paper package。
```

Colab workflow 已按以下顺序接入:

```text
runtime detection
validation / pilot_paper internal ablation
statistical confidence interval
validation artifact rebuild dry-run
validation_scale gate
package to Google Drive
```

因此, 后续真实复跑可以延后; 在复跑发生后, 新增后处理会自动把 validation_scale 缺口显式落盘, 而不是让缺口停留在人工判断。


## 2026-06-23 external_baseline adapter comparison 接入

Colab workflow 已在 runtime detection 之后新增 external baseline comparison 步骤:

```text
runtime detection
external_baseline adapter comparison
small-scale mechanism pilot check
validation / pilot_paper internal ablation
statistical confidence interval
validation artifact rebuild dry-run
validation_scale gate
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



### external_baseline adapter comparison 运行语义

该步骤已经作为 generative video model probe 的标准后处理步骤接入。其语义为:

```text
input: runtime detection records + callback trajectory records
adapter_boundary: external_baseline/primary/<baseline_id>/adapter
scheduler: experiments/generative_video_model_probe.external_baseline_runner
output: external_baseline_score_records + comparison table + decision + report
claim_boundary: proxy comparison only until modern baseline adapters produce measured records
```

该步骤必须位于 runtime detection 之后, 因为 adapter 需要读取真实 attacked video detection 链路产生的 governed records。该步骤必须位于 validation_scale gate 和 pilot_paper gate 之前: validation_scale gate 需要检查 `validation_external_baseline_comparison_records_ready`, pilot_paper gate 需要检查同批 held-out test trace 的 `pilot_paper_external_baseline_comparison_ready`。

## 2026-06-24 validation_scale hard-blocker runner workflow 接入

当前 validation_scale notebook workflow 已接入两个硬阻断 runner:

```text
experiments.generative_video_model_probe.adaptive_attack_runner
experiments.generative_video_model_probe.claim3_downgrade
```

运行顺序为:

```text
runtime detection
external_baseline adapter comparison
small_scale_mechanism_pilot_check
validation_internal_ablation
adaptive_attack validation proxy
Claim-3 downgrade gate
statistical_confidence_interval
validation_artifact_rebuild_dry_run
validation_scale_gate
package
```

其中 adaptive attack runner 只提供 validation proxy records, 不支撑 full_paper adaptive robustness claim。Claim-3 downgrade gate 只允许 validation_scale 合规继续, 不替代最终必须实现的 replay/sketch gate。

新增落盘产物包括:

```text
records/adaptive_attack_records.jsonl
tables/adaptive_attack_table.csv
artifacts/adaptive_attack_decision.json
reports/adaptive_attack_report.md
records/claim3_downgrade_records.jsonl
tables/claim3_downgrade_table.csv
artifacts/claim3_downgrade_decision.json
reports/claim3_downgrade_report.md
```

## 2026-06-24 replay/sketch gate validation proxy 接入

validation_scale workflow 已新增 replay/sketch gate 步骤, 位于 adaptive attack proxy 与 Claim-3 downgrade gate 之间:

```text
runtime detection
external_baseline adapter comparison
small_scale_mechanism_pilot_check
validation_internal_ablation
adaptive_attack validation proxy
replay/sketch gate validation proxy
Claim-3 downgrade gate
statistical_confidence_interval
validation_artifact_rebuild_dry_run
validation_scale_gate
package
```

新增步骤写出 trajectory sketch verification、replay uncertainty、wrong sampler replay 和 wrong prompt replay 四类 governed records。`validation_artifact_rebuild_dry_run` 与 Google Drive package manifest 已纳入该步骤产物。当前该步骤只解除 validation_scale 工程入口缺口, 不解除 full_paper Claim-3 强支持阻塞。


## 2026-06-24 pilot_paper FPR=0.01 工程入口

当前 `generative_video_model_probe` 已新增 `pilot_paper` 语义层级, 用于在 validation_scale 通过并生成 validation_scale_to_pilot_paper_transition_decision 后执行小样本论文级结果包。该层级不是 workflow-only pilot, 而是小规模跑代表性 paper 协议并产出 pilot 级论文结果。

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
paper_protocol_difference_from_full_paper: sample_scale_target_fpr_and_attack_coverage
prompt_count: 21
seed_per_prompt: 8
calibration_seed_per_prompt: 4
test_seed_per_prompt: 4
unique_video_count: 168
calibration_unique_video_count: 84
test_unique_video_count: 84
expected_calibration_negative_event_count: 1008
expected_heldout_test_negative_event_count: 1008
expected_heldout_attacked_positive_event_count: 252
target_fpr: 0.01
threshold_protocol: calibration_split_to_frozen_threshold_to_heldout_test_split
```

该阶段通过后允许报告 `pilot_paper_calibrated_heldout_claim_ready` 和 pilot_paper 级 `TPR@FPR=0.01` 论文主张。通过条件不仅包括 calibration / held-out threshold 协议, 还包括 external_baseline comparison 和内部消融矩阵对同批 held-out test trace 的覆盖。它与 full_paper 的区别只在样本规模和统计置信度, 但仍不允许报告 `TPR@FPR=0.001` 或 full_paper 规模主表结论。


### 2.11 pilot_paper gate 前置 baseline 与内部消融闭环

在进入真实 `PROFILE = pilot_paper` 复跑前, baseline 和内部消融必须已经在 `validation_scale` 中作为完整机制门禁闭合。`pilot_paper_gate` 仍会复核同批 held-out test trace 的 records 覆盖, 但不应承担补 baseline 或补消融的职责。它会读取:

```text
artifacts/external_baseline_comparison_decision.json
records/external_baseline_score_records.jsonl
artifacts/validation_internal_ablation_decision.json
records/validation_internal_ablation_records.jsonl
```

必须满足:

```text
pilot_paper_external_baseline_comparison_ready == true
pilot_paper_internal_ablation_matrix_ready == true
minimum_external_baseline_measured_adapter_count >= 7
minimum_modern_external_baseline_formal_adapter_count >= 5
minimum_internal_ablation_variant_count >= 8
pilot_paper_external_baseline_trace_count_min >= 84
pilot_paper_internal_ablation_trace_count_min >= 84
```

这一实现属于项目特定 gate 设计: 它强制 `validation_scale` 先闭合对比链路和消融链路, 再允许 `pilot_paper` 输出 pilot 级 fixed-FPR 论文主张。显式 DTW 与 frame matching 仍只是同步 control proxy, 现代视频水印 baseline 的正式 adapter 必须在 `validation_scale` 通过前基于项目内 clone / build / run / adapt / record 产出 `measured_formal` records。


### 2.12 现代视频水印 baseline 正式 adapter 要求

`pilot_paper` 与后续更大规模 paper 运行的差异必须由 protocol config 显式记录, 不能变成 baseline 协议缺口。因此 `validation_scale` 通过前必须完成完整现代 baseline 自包含执行链路, 并在小样本上通过项目内 clone / build / run / adapt / record 真实产出 comparison records。当前工程已接入以下正式 adapter 边界:

```text
videoshield
vidsig
videoseal
```

每个现代 baseline 必须配置对应环境变量命令:

```text
SSTW_VIDEOSHIELD_EVAL_COMMAND
SSTW_VIDSIG_EVAL_COMMAND
SSTW_VIDEOSEAL_EVAL_COMMAND
```

命令未配置时, adapter 会写 unsupported record, `validation_scale` 必须阻断进入 `pilot_paper`; 若仍进入 `pilot_paper`, `pilot_paper` gate 也会因为 `missing_modern_external_baseline_formal_adapter_names` 失败。这是硬阻断, 不是 warning。

### 2.13 profile-driven Notebook 重构状态

当前 Colab workflow 已从单一综合 Notebook 拆分为 runtime、5 个主实验 baseline formal reference、
formal scoring、paper evidence postprocess 和 paper gate 等职责明确入口, 并由统一配置控制 profile 切换:

```text
configs/paper_workflow/generative_video_notebook_workflows.json
```

推荐执行职责为:

```text
motion_threshold_calibration_colab.ipynb: 只运行 motion calibration split 并冻结 threshold artifact
generative_video_runtime_colab.ipynb: 运行 Wan2.1 生成、formal metrics、motion threshold 复用、attack 和 detection
5 个主实验 modern external baseline formal reference Notebook: 分别运行对应 baseline 的官方流程并生成项目内 official bundle, 不默认调用全量 runner 转写 measured_formal records
formal_comparison_scoring_colab.ipynb: 恢复 5 个主实验 official reference 阶段包后运行全量 external baseline comparison、self-containment、公平校准和差值区间统计
paper_evidence_postprocess_colab.ipynb: 恢复 runtime、motion threshold 和 formal comparison scoring 阶段包后运行 internal ablation、adaptive attack、replay/sketch 或 Claim-3 downgrade、CI、低 FPR 阻断记录和数据切分泄漏检查
paper_gate_and_package_colab.ipynb: 恢复 runtime、motion threshold、formal comparison scoring 和 paper evidence postprocess 阶段包后只运行最终 gate、transition、artifact rebuild dry run、figure/package manifest 和 package
```

旧的通用 external baseline scoring Notebook 已删除。validation-scale 推荐主流程
保留 5 个 baseline 专用 official reference Notebook、独立
`formal_comparison_scoring_colab.ipynb`、`paper_evidence_postprocess_colab.ipynb` 和
`paper_gate_and_package_colab.ipynb` 的最终聚合门禁。

切换运行层级时只应修改环境变量或配置:

```text
SSTW_WORKFLOW_PROFILE=validation_scale
SSTW_WORKFLOW_PROFILE=pilot_paper
```

不应在 Notebook 中复制新的 Google Drive 目录、样本上限或 profile 分支。未来 `full_paper` 已在配置中登记, 但状态为 `design_registered_not_ready`, 当前不允许真实运行或支撑 claim。

`motion_threshold_artifact_run_root_relative` 指向独立 calibration run_root。该设计允许 `validation_scale` 与 `pilot_paper` 使用各自隔离的 evaluation run_root, 同时复用同一个已冻结 motion threshold artifact, 防止把 calibration 输出与 evaluation 输出混写。
