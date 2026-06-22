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

### 1.5 默认外部 baseline

```text
explicit_dtw_temporal_alignment
frame_matching_temporal_registration
```

项目默认不保留 VideoSeal、RivaGAN、VidStamp 作为本阶段外部 baseline, 以降低与并行论文或既有实验叙事的重复风险。

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
generic_ssm_baseline
```

### 1.7 通过标准

1. 主线记录必须说明 generation model 是 Flow Matching / velocity-field sampler 相关模型。
2. velocity / flow trajectory proxy 必须参与同步证据。
3. SSTW full method 在 `TPR@FPR=0.01` 下优于外部 baseline 与内部机制 baseline。
4. 质量、运动和语义指标不显示不可接受退化。

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
