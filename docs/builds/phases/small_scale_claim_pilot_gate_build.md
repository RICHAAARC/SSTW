# small_scale_mechanism_pilot_check 分阶段构建流程

本文档保留旧文件名 `small_scale_claim_pilot_gate_build.md`, 但当前语义已经调整为 `small_scale_mechanism_pilot_check`。该检查不再作为主干门禁, 只作为 `mechanism_validation` 下的小样本机制检查记录。本文档只描述工程、协议、records 和 artifact 状态, 不直接支撑论文最终 claim。当前项目已额外引入 `pilot_paper` 作为小样本论文级结果层级; 二者不能混用。

## 1. 本阶段构建流程

### 1.1 阶段目标

该检查在进入 `probe_paper` 前, 以较小成本验证主要 claim 是否有成立迹象。该检查不产生论文级结果表格, 也不能直接决定是否进入 `pilot_paper` 或 `full_paper`。`small_scale_mechanism_pilot_check` 通过后只允许进入 `pilot_paper`; `probe_paper` 通过并生成 `probe_paper_to_pilot_paper_transition_decision` 后, 才允许进入 `pilot_paper`; `probe_paper` 通过并生成 `probe_paper_to_pilot_paper_transition_decision` 后, 才允许进入 `pilot_paper`。

### 1.2 建议规模

```text
N_prompt >= 8
N_seed_per_prompt >= 2
N_attack >= 3
N_calibration_negative_family >= 4
N_method_variant >= 6
```

### 1.3 必须覆盖的攻击和错配

```text
video_compression
temporal_crop
frame_rate_resampling
vae_reencode_attack
wrong_sampler_replay
wrong_key_control
```

### 1.4 必须记录字段

```text
negative_family
flow_velocity_alignment_gain
path_marginal_gain_at_fixed_fpr
trajectory_payload_redundancy
quality_guard_violation_rate
negative_tail_status
wrong_key_score_separation_passed
wrong_sampler_replay_control_not_equivalent
claim_support_status
```

### 1.5 通过标准

```text
trajectory_trace_capture_success_rate >= 0.95
flow_velocity_alignment_gain > 0
path_marginal_gain_at_fixed_fpr > 0
trajectory_payload_redundancy <= preset_limit
quality_guard_violation_rate <= preset_limit
negative_tail_inflation_not_detected = true
wrong_key_score_separation_passed = true
wrong_sampler_replay_control_not_equivalent = true
```

### 1.6 进入 probe_paper 的升级条件

`small_scale_mechanism_pilot_check` 通过只表示可以进入 `probe_paper` 构建, 不表示可以进入 `pilot_paper` 或 `full_paper`。进入 `probe_paper` 还必须确认:

```text
all_pilot_records_are_governed = true
proxy_only_records_are_labeled = true
runtime_attack_detection_chain_ready = true
modern_baseline_plan_exists = true
adaptive_attack_plan_exists = true
replay_or_claim3_downgrade_plan_exists = true
artifact_rebuild_dry_run_plan_exists = true
```

若上述任一条件不满足, `small_scale_mechanism_pilot_check` 即使分数通过, 也只能继续补齐工程流程并进入 `probe_paper`, 不能直接进入 `pilot_paper` 或 `full_paper`。

## 2. 当前阶段完成情况

### 2.1 当前阶段判定

`small_scale_mechanism_pilot_check` 当前判定为:

```text
structure_ready / protocol_ready / external_validation_required
```

该阶段现在已经满足进入条件, 因为前置阶段已经完成:

```text
flow_model_adapter_preflight: PASS
sampling_time_constraint_probe smoke: PASS
sampling_time_constraint_probe recommended: PASS
motion_threshold_calibration: PASS
```

最新 sampling-time recommended 证据:

```text
package_batch_id: 20260618_023447_f325e2a5
implementation_evidence_status: PASS
mechanism_evidence_status: PASS
missing_mechanism_requirements: []
```

### 2.2 pilot 必须回答的问题

pilot 不能重复证明 callback 是否能工作, 而应验证机制证据是否值得扩展到 `probe_paper`、`pilot_paper` 和 `full_paper` 主干阶段。必须重点检查:

```text
path_marginal_gain_at_fixed_fpr > 0
negative tail 没有膨胀
wrong_sampler_replay 不能伪造正确轨迹
wrong_key / without-key control 保持分离
quality_guard 通过
attack 后 trajectory evidence 与 endpoint evidence 不发生系统性冲突
```

### 2.3 下一步建议

当前 Colab 入口已经切换到 small-scale pilot 参数:

```text
notebook: paper_workflow/colab_notebooks/generative_video_generation_colab.ipynb -> paper_workflow/colab_notebooks/generative_video_quality_scoring_colab.ipynb -> paper_workflow/colab_notebooks/sstw_mechanism_postprocess_colab.ipynb -> paper_workflow/colab_notebooks/runtime_attack_colab.ipynb -> paper_workflow/colab_notebooks/runtime_detection_colab.ipynb
PROFILE: pilot
MODEL_ID: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
CROSS_MODEL_ID: empty
prompt_limit: 8
seed_limit: 2
num_inference_steps: 16
num_frames: 49
height: 320
width: 512
run_cross_model: false
```

该设置会优先验证主模型上的 pilot 规模生成链路。attack matrix、negative family 和 wrong-sampler replay 仍需要由后续 pilot postprocess / checker 或专门 runner 继续闭合。

下一步建议按以下目标规模补齐 pilot:

```text
8 prompts
2 seeds per prompt
3 attacks
4 negative families
6 method variants
```

`small_scale_mechanism_pilot_check` 失败时不得进入 `probe_paper`。`small_scale_mechanism_pilot_check` 通过后, 只允许进入 `probe_paper` 的小样本论文闭合验证, 不允许直接进入 `pilot_paper` 或 `full_paper`。

### 2.4 motion threshold calibration 冻结策略

当前 small-scale claim pilot 可以使用已经通过的 engineering calibration threshold。pilot profile 不得用 16 条 pilot 样本重新估计 motion threshold, 否则会把独立 calibration split 覆盖为 `INSUFFICIENT_SAMPLE`。

```text
motion_threshold_calibration_decision: PASS
motion_delta_threshold: 0.010607
threshold_id: motion_delta_calibrated_v1
threshold_source_split: calibration
usage: frozen_engineering_motion_threshold_for_small_scale_pilot
```

该阈值用于 pilot 阶段阻止明显低运动视频支撑 motion-related claim。它可以解除 `blocked_until_motion_threshold_calibration`, 但仍不等价于论文级 `TPR@FPR=0.01` 或 `TPR@FPR=0.001` 证据。最终 paper claim 前仍需要更大 held-out negative split 和 frozen fixed-FPR 评估。

pilot 报告中应使用以下边界表达:

```text
mechanism proxy evidence can support small-scale pilot progression.
formal motion gate uses frozen engineering calibration threshold.
paper-level fixed-FPR evidence remains required before final claim.
```

### 2.5 pilot gate 自动审计器

当前已新增 small-scale claim pilot gate 自动审计入口:

```text
experiments/generative_video_model_probe/pilot_claim_gate.py
scripts/check_results/small_scale_claim_pilot_result_checker.py
```

该 checker 会从已有 governed records 中自动审计:

```text
prompt_count
seed_per_prompt_min
attack_count
negative_family_count
method_variant_count
path_marginal_gain_at_fixed_fpr
negative_tail_status
wrong_key_score_separation_passed
wrong_sampler_replay_control_not_equivalent
replay_uncertainty_mean
motion_threshold_calibration_required
```

当前 Google Drive pilot run 的 dry-run 结论为:

```text
pilot_gate_decision: FAIL
claim_support_status: workflow_progression_only
prompt_count: 8
seed_per_prompt_min: 2
quality_motion_semantic_proxy_pass: true
formal_motion_claim_status: blocked_by_formal_motion_consistency
motion_threshold_source_split: heuristic_precalibration
```

当前缺口由 checker 自动报告为:

```text
attack_matrix_ready
negative_family_ready
method_variant_ready
path_marginal_gain_ready
negative_tail_ready
wrong_key_separation_ready
wrong_sampler_replay_ready
replay_uncertainty_ready
```

这说明当前 Wan2.1 pilot 生成链路和 proxy 后处理可继续支撑 workflow progression, 但仍不能进入 `full_paper` 或最终论文 claim。下一步应补齐 pilot postprocess / runner, 产生 attack、negative family、wrong-sampler replay 和 replay uncertainty governed records。

### 2.6 pilot matrix proxy postprocess 状态

已新增 small-scale claim pilot matrix proxy 后处理入口:

```text
experiments/generative_video_model_probe/pilot_matrix_postprocess.py
```

该 runner 基于现有 generation records 与 trajectory records 构造受治理的 proxy matrix records, 覆盖:

```text
3 attacks
4 negative families
6 method variants
wrong_key separation
wrong_sampler_replay control
path_marginal_gain_at_fixed_fpr
negative_tail_status
replay_uncertainty_mean
```

当前 Google Drive pilot run 已写出:

```text
records/small_scale_claim_pilot_matrix_records.jsonl
tables/small_scale_claim_pilot_matrix_table.csv
artifacts/small_scale_claim_pilot_matrix_decision.json
reports/small_scale_claim_pilot_matrix_report.md
```

当前自动审计结果更新为:

```text
pilot_matrix_postprocess_decision: PASS
pilot_matrix_record_count: 480
pilot_matrix_attack_count: 3
pilot_matrix_method_variant_count: 6
pilot_matrix_negative_family_count: 4
path_marginal_gain_at_fixed_fpr: 0.075
replay_uncertainty_mean: 0.073608
missing_pilot_requirements: []
claim_support_status: blocked_until_motion_threshold_calibration
pilot_gate_decision: FAIL
```

该结果表示 pilot 矩阵的 proxy 后处理记录已经补齐, 但仍不能进入 `full_paper` 或最终论文 claim。阻塞原因不再是矩阵缺失, 而是 formal motion gate 仍使用 `heuristic_precalibration` 阈值, 需要后续 `motion_threshold_calibration` 阶段。

### 2.7 runtime video-file attack runner 状态

已新增并运行真实文件级 runtime attack runner:

```text
experiments/generative_video_model_probe/attack_runner.py
```

该 runner 对已有 Wan2.1 pilot 生成视频执行实际 mp4 文件级攻击, 并写出 attacked videos 与 governed records:

```text
records/runtime_attack_records.jsonl
tables/runtime_attack_table.csv
artifacts/runtime_attack_decision.json
reports/runtime_attack_report.md
attacked_videos/
```

当前 Google Drive pilot run 的 runtime attack 结果为:

```text
runtime_attack_decision: PASS
runtime_attack_record_count: 48
runtime_attack_ready_count: 48
runtime_attack_count: 3
attack_matrix_evidence_level: runtime_video_file
claim_support_status: runtime_attack_evidence_only
```

当前 small-scale pilot gate 更新为:

```text
missing_pilot_requirements: []
attack_count: 6
runtime_attack_record_count: 48
runtime_attack_ready_count: 48
pilot_gate_decision: FAIL
claim_support_status: blocked_until_motion_threshold_calibration
```

该结果表示工程层面的 runtime attack 链路已经闭合, 但仍不能进入 final claim。剩余阻塞是 `motion_threshold_calibration`, 以及后续如需论文级攻击结论, 仍需要把 runtime attacked videos 接入正式 detection / scoring, 而不是只依赖 proxy matrix score。

### 2.8 runtime attacked video detection 闭环状态

已新增 runtime attacked video detection runner:

```text
experiments/generative_video_model_probe/detection_runner.py
```

该 runner 读取:

```text
records/runtime_attack_records.jsonl
attacked_videos/*.mp4
records/trajectory_trace.jsonl
```

并写出:

```text
records/runtime_detection_records.jsonl
tables/runtime_detection_table.csv
artifacts/runtime_detection_decision.json
reports/runtime_detection_report.md
```

当前 Google Drive pilot run 的 runtime detection 结果为:

```text
runtime_detection_decision: PASS
runtime_detection_record_count: 48
runtime_detection_ready_count: 48
runtime_detection_detectable_count: 48
runtime_detection_attack_count: 3
runtime_detection_score_mean: 0.781174
claim_support_status: runtime_detection_evidence_only
```

pilot gate 现在会显式检查: 如果存在 ready runtime attack records, 则必须存在对应 ready runtime detection records。该规则用于防止只生成 attacked videos 而未进入 detection scoring 的半闭环状态。

历史 pilot gate 在 motion calibration 未完成时仍为:

```text
pilot_gate_decision: FAIL
claim_support_status: blocked_until_motion_threshold_calibration
missing_pilot_requirements: []
```

该历史结论表示 pilot 的工程矩阵与 runtime attack / detection 链路已经闭合, 当时剩余阻塞项不是工程链路缺失, 而是 motion threshold calibration 尚未完成。最新 motion calibration 已经 PASS, 因此应重新运行 pilot profile, 让 gate 使用冻结 calibration artifact 重新计算。

### 2.9 2026-06-23 切换记录: small-scale pilot profile

Colab 入口已切换为:

```text
PROFILE: pilot
MODEL_ID: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

执行顺序为:

```text
prepare prompt suite
Wan2.1 generation with PROFILE = pilot
formal_metric_runner
reuse existing motion_threshold_calibration_decision.json
mechanism postprocess
pilot matrix postprocess
runtime attack
runtime detection
small-scale claim pilot gate
package_outputs
```

其中 `reuse existing motion_threshold_calibration_decision.json` 是关键约束。pilot profile 只读取已经通过的 calibration artifact, 不会重新运行 `motion_threshold_calibration`。

### 2.10 formal motion 失败样本的 claim 过滤修复

最新 small-scale pilot run 中发现 1 个正向运动样本未通过 formal motion consistency:

```text
prompt_id: heldout_rotation_scene
seed_id: seed_main_b
trajectory_trace_id: trace_0005
formal_motion_gate_failure_reason: motion_delta_below_min
formal_metric_blocking_reason: formal_motion_consistency_not_ready
```

该样本的视觉质量和语义一致性可以通过, 但实际生成结果接近低运动状态, 因此不能支撑 velocity / trajectory / motion 相关 claim。治理处理方式为:

```text
保留 generation record
保留 formal metric record
不物理删除视频文件
从 motion claim eligible set 中排除
从 pilot matrix proxy records 中排除
在 small-scale pilot gate 中计入 formal_motion_claim_ready 缺口
```

本次修复新增公共筛选模块:

```text
experiments/generative_video_model_probe/formal_motion_claim_filter.py
```

下游 runner 和 gate 现在遵循同一规则:

```text
污染或剔除判断不读取 S_final、S_final_conservative 或任何最终检测分数。
筛选只依赖 formal_visual_quality_ready、formal_motion_consistency_ready、formal_semantic_consistency_ready 和 motion_claim_role。
```

修复后当前 Google Drive run 的派生 artifacts 已重新写出:

```text
motion_claim_eligible_generation_count: 15
motion_claim_excluded_generation_count: 1
pilot_matrix_record_count: 450
formal_motion_claim_status: blocked_by_formal_motion_consistency
pilot_gate_decision: FAIL
claim_support_status: workflow_progression_only
missing_pilot_requirements:
  - seed_coverage_ready
  - formal_motion_claim_ready
```

该历史结果表示当时 16 条生成记录中只有 15 条可用于 motion claim。由于 `heldout_rotation_scene` 只剩 1 个合格 seed, 当时 pilot 的 seed 覆盖率不足, 因此不能进入 `full_paper`。该阻塞已由后续原生复跑解除。

### 2.11 heldout pilot prompt 可观测运动修复

为修复 `heldout_rotation_scene / seed_main_b` 在真实 Wan2.1 pilot 中生成低运动结果的问题, 已更新 prompt suite 输入设计。旧 prompt:

```text
A blue cube rotates gently on a plain gray surface with soft shadows and smooth motion.
motion_pattern_id: gentle_rotation
```

已替换为更强可观测运动的 heldout prompt:

```text
A large blue cube with bright orange arrow markings slides from the far left edge to the far right edge while spinning rapidly for a full rotation on a plain gray floor, fixed camera, the cube fills at least one third of the image, strong visible displacement in every frame.
motion_pattern_id: large_rotation_translation
motion_claim_role: positive_motion
```

同时, pilot 主 prompt 已显式登记:

```text
motion_claim_role: positive_motion
```

该变更的作用是提高后续 `PROFILE = pilot` 运行中 positive motion 样本的可观测运动概率。它不会修改旧 run 的事实记录, 也不会把已经失败的样本重新解释为通过。该旧 run 在当时仍保持:

```text
pilot_gate_decision: FAIL
missing_pilot_requirements:
  - seed_coverage_ready
  - formal_motion_claim_ready
```

本地 Google Drive 同步目录中的 prompt suite 已更新:

```text
G:\我的云端硬盘\SSTW\datasets\generative_video_prompt_suite\prompt_seed_suite.json
prompt_suite_id: generative_video_probe_prompt_suite_motion_observability_and_pilot_repair
```

后续已经重新执行 Colab `PROFILE = pilot`, 并产生新的 generation records、formal records 和 pilot gate artifacts。最新结论见本文档末尾的“2026-06-23 最新原生复跑状态”。


## 3. 当前查漏补缺状态

| 项目 | 当前标注 |
|---|---|
| 完成状态 | 已完成 small-scale pilot, 可进入 probe_paper |
| 主要差距项 | 本阶段阻塞已解除; 剩余差距转移到 probe_paper、pilot_paper、现代外部 baseline 主表对比、内部消融和论文级 fixed-FPR。 |
| 下一步构建方向 | 进入 probe_paper, 并在 probe_paper 通过后生成 probe_paper_to_pilot_paper_transition_decision 再进入 pilot_paper, 同步补现代外部 baseline、内部消融和 CI reporter。 |
| full_paper 影响 | 未满足本阶段要求时, 不得把相关结果写入 full_paper supported claim。 |

### 3.1 快速检查清单

```text
stage_status: 已完成 small-scale pilot, 可进入 probe_paper
gap_item: 本阶段阻塞已解除; 剩余差距转移到 probe_paper、pilot_paper、现代外部 baseline 主表对比、内部消融和论文级 fixed-FPR。
next_action: 进入 probe_paper, 并在 probe_paper 通过后生成 probe_paper_to_pilot_paper_transition_decision 再进入 pilot_paper, 同步补现代外部 baseline、内部消融和 CI reporter。
full_paper_blocking_rule: unresolved_gap_blocks_full_paper_claim
```

## 4. small-scale pilot 结果使用边界

`small_scale_mechanism_pilot_check` 的作用是判断是否值得进入 probe_paper, 不是产出论文主结果, 也不是进入 pilot_paper 的直接门票。pilot_paper 是后续独立结果层级, 必须使用 calibration / held-out split 和 frozen threshold artifact。即使 `small_scale_mechanism_pilot_check` PASS, 也必须遵守:

```text
pilot_records_not_used_for_main_detection_table
pilot_thresholds_not_used_for_full_paper_thresholds
pilot_prompt_fixes_not_applied_retroactively
pilot_baseline_results_not_used_as_external_main_comparison
pilot_attack_results_not_used_as_full_robustness_claim
```

`small_scale_mechanism_pilot_check` PASS 后的下一步只能是 probe_paper, 不是 pilot_paper 或 full_paper。probe_paper 通过后, 只能通过 `probe_paper_to_pilot_paper_transition_decision` 进入 pilot_paper。若 Codex 检测到用户或脚本试图从 `small_scale_mechanism_pilot_check` 直接生成 full_paper package, 应将其标记为阶段跳跃。

## 5. small_scale_mechanism_pilot_check 通过后的 probe_paper 准入清单

`small_scale_mechanism_pilot_check` 通过后, 启动 probe_paper 前仍需确认:

```text
probe_paper_prompt_manifest_prepared
probe_paper_seed_plan_prepared
probe_paper_attack_manifest_prepared
probe_paper_baseline_manifest_prepared
probe_paper_ablation_manifest_prepared
probe_paper_replay_or_sketch_plan_prepared
probe_paper_artifact_rebuild_plan_prepared
probe_paper_resource_budget_checked
```

上述条件缺失时, 只能补齐 probe_paper 规划, 不能进入 `pilot_paper` 或 `full_paper`。


## 2026-06-23 最新原生复跑状态

最新 Wan2.1 `PROFILE = pilot` 原生复跑已经解除本阶段阻塞:

```text
small_scale_pilot_gate_decision: PASS
claim_support_status: supported_by_small_scale_claim_pilot_records
motion_claim_eligible_generation_count: 16
motion_claim_excluded_generation_count: 0
seed_per_prompt_min: 2
runtime_attack_ready_count: 48
runtime_detection_ready_count: 48
protocol_missing_failures: []
```

本阶段现在的结论是: 可以进入 probe_paper 小样本论文闭合验证。该结论不能被解释为 full_paper ready, 也不能用于生成最终 `TPR@FPR=0.001` 主表。

## 2026-06-24 formal motion exclusion gate 判定修复

本次修复明确区分“样本级 formal motion exclusion”和“阶段级 pilot gate 阻断”。当某个 positive motion 样本未通过 formal motion consistency 时, 该样本仍必须保留 generation record 与 formal metric record, 但不能进入 motion / trajectory claim 的 eligible set。

修复后的阶段级判定规则为:

```text
formal_motion_claim_status = ready
  表示所有正向 motion 样本均可用于 claim。

formal_motion_claim_status = ready_with_formal_motion_exclusions
  表示存在样本被 formal gate 剔除, 但剩余 eligible 样本仍可用于 claim 覆盖率统计。
  是否可继续推进由 prompt_count、seed_per_prompt_min、attack_count、negative_family_count、method_variant_count 等覆盖规则决定。

formal_motion_claim_status = blocked_by_formal_motion_consistency
  仅表示没有足够 positive motion eligible records 支撑 motion claim。
```

该修复不依赖 `S_final`、`S_final_conservative` 或任何最终检测判定分数, 因此不违反污染过滤约束。它只改变 gate 对已被剔除样本的阶段级解释, 不改变样本级过滤逻辑。对于 probe_paper 风格的 8 prompt × 3 seed 运行, 如果 1 个样本被剔除后仍保持 `seed_per_prompt_min >= 2`, pilot gate 可以继续通过并把剔除数量写入审计字段。
