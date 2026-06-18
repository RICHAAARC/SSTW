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

未完成 calibration 时, 当前 formal motion gate 只能解释为:

```text
threshold_id: motion_delta_heuristic_v1
threshold_source_split: heuristic_precalibration
usage: pilot_guardrail
```

禁止将 evaluation split 或每次生成后的结果反向用于动态调整阈值。正式实验必须使用预先校准并冻结的 threshold artifact。

