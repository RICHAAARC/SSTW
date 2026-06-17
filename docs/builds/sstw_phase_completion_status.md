# SSTW 分阶段完成情况记录

## 0. 文档定位

本文档用于记录 SSTW 各阶段的当前完成情况。它与 `docs/builds/sstw_project_construction_flow.md` 的职责不同: 整体构建流程文档只定义项目应如何构建, 本文档记录每个阶段在当前仓库中的工程状态、证据状态和后续缺口。

本文档不直接支撑论文 supported claims。论文 claims 必须由 governed records、tables、figures、reports 或 manifests 支撑。

## 1. 阶段状态分级

“当前状态”字段可以包含一个或多个状态值。多个状态值用 `/` 连接, 表示该阶段同时满足多个条件。状态值不是论文结论, 也不是投稿可用 claim; 它只描述当前仓库在工程、协议、artifact 或外部验证方面的准备程度。

| 状态值 | 具体含义 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| `structure_ready` | 阶段所需的目录、配置、runner、checker、packager、文档或基础模块已经建立。 | 可以说明该阶段具备继续实现和运行的工程入口。 | 不能说明核心算法机制已经有效, 也不能说明实验结果已经成立。 |
| `mechanism_ready` | 阶段核心机制已有可运行或可测试实现, 例如 state inference、trajectory observation、VAE transfer 或 sampling constraint 的核心函数和 runner 已存在。 | 可以说明该阶段具备机制级测试或消融的代码基础。 | 不能说明真实模型、真实数据或 fixed-FPR 结果已经达到论文目标。 |
| `protocol_ready` | 阶段已有固定 split、sample role、threshold、baseline、control、checker 或 manifest 约束。 | 可以说明该阶段的实验不应再随意改变校准、阈值、样本角色和 baseline 口径。 | 不能说明协议下已经产出充分结果, 也不能说明所有 claim 已经被 governed artifacts 支撑。 |
| `artifact_ready` | 阶段已有从 records 重建 tables、figures、reports、packages 或 manifests 的入口。 | 可以说明该阶段具备 artifact rebuild 的工程通道。 | 不能说明 artifacts 中已有最终论文结果, 也不能说明可手工补写结果。 |
| `external_validation_required` | 阶段需要 Colab、GPU、真实视频模型、真实 VAE 链路或外部 baseline 运行结果继续验证。 | 可以说明该阶段存在本地 CPU / synthetic 结果无法覆盖的证据缺口。 | 不能说明阶段失败; 它只表示需要外部运行环境或真实模型结果补齐证据。 |

### 1.1 状态组合解释

常见组合含义如下:

| 状态组合 | 含义 |
|---|---|
| `structure_ready / protocol_ready` | 工程入口和实验协议已经建立, 但核心机制结果或真实模型验证仍需继续补齐。 |
| `structure_ready / mechanism_ready` | 工程入口和核心机制代码已经建立, 但协议完整性、artifact rebuild 或外部验证可能仍需补强。 |
| `structure_ready / artifact_ready` | 工程入口和 artifact rebuild 入口已经建立, 但 artifacts 是否能支撑论文 claim 取决于上游 records 是否充分。 |
| `structure_ready / protocol_ready / external_validation_required` | 本地流程和协议已经准备好, 下一步重点是使用 Colab / GPU / 真实模型产出 governed records。 |

### 1.2 使用规则

1. `structure_ready` 是最低工程状态, 只代表“可以继续推进”, 不代表“机制成立”。
2. `mechanism_ready` 必须对应仓库中的可运行或可测试模块, 不能只由文档描述支撑。
3. `protocol_ready` 必须对应固定 split、threshold、baseline、control、checker 或 manifest 规则。
4. `artifact_ready` 必须对应可由 records / manifests 重建产物的脚本或 runner。
5. `external_validation_required` 表示需要外部真实运行补证, 不应被解释为已通过或未通过。
6. 任何状态值都不能直接支撑论文 supported claim; supported claim 必须映射到 governed records、tables、figures、reports 或 manifests。

## 2. 阶段完成情况总览

| 阶段 | 当前状态 | 主要依据 | 后续重点 |
|---|---|---|---|
| protocol_governance_foundation | structure_ready / protocol_ready | configs、field_registry、harness、tests/constraints | 继续随新增字段、negative family 和旧字段映射同步注册 |
| synthetic_state_inference_sanity | structure_ready / mechanism_ready | experiments/synthetic_state_inference 与 state_space_watermark 模块 | 保持与新 trajectory / replay / negative family 字段对齐 |
| real_video_latent_transfer_check | structure_ready / mechanism_ready | experiments/real_video_latent_transfer 与 main/vae | 继续验证 VAE 链路低误报与 endpoint consistency |
| state_space_inference_formalization | structure_ready / mechanism_ready | experiments/state_space_formalization 与 state-space 模块拆分 | 补强 key-conditioned、admissibility、negative tail 和 generic SSM 消融 |
| trajectory_observation_core_probe | structure_ready / mechanism_ready | main/trajectory 与 trajectory_observation_core runner | 补强 time-reparameterization invariance、path marginal gain 与 wrong sampler replay control |
| flow_model_adapter_preflight | structure_ready / external_validation_required | generation model registry、scheduler adapter、latent capture、trajectory capture、Colab runtime | 在 Wan2.1 / L4 / Colab 环境验证 callback、time grid、sampler signature 与 velocity / displacement proxy |
| sampling_time_constraint_probe | structure_ready / protocol_ready / external_validation_required | sampling constraint adapter、Colab runtime、checker、packager | 验证 velocity / flow trajectory 确实参与同步, 并与 endpoint-aware control 对齐 |
| small_scale_claim_pilot_gate | structure_ready / protocol_ready / external_validation_required | sampling-time 与 generative probe runner、external baseline runner、checker | 以 pilot split 验证 Claim-1、Claim-2 和部分 Claim-3 是否值得进入 full generation |
| generative_video_model_probe | structure_ready / protocol_ready / external_validation_required | generative probe runner、external baseline runner、Colab 入口 | 在 pilot gate 后以 Wan2.1 L4 / Colab 真实结果验证主 claim |
| replay_and_authenticated_sketch_gate | structure_ready | digest、manifest、trajectory trace 基础模块 | 补齐 authenticated sketch、replay uncertainty、wrong prompt replay 与 checker |
| flow_specific_adaptive_attack_gate | structure_ready | attacks、trajectory controls、generative probe attack runner | 补强 scheduler change、time grid jitter、endpoint-path decoupling 等 Flow-specific adaptive attacks |
| submission_package_freeze | structure_ready / artifact_ready | submission_freeze_preparation runner、main_tables、readiness_summary | 等待真实 GPU records、pilot gate records、negative family records 后重建论文 artifacts |

## 3. 当前不应越界的结论

1. 若缺少 Wan2.1 真实 GPU records, 不应宣称完整真实模型主结果已经产出。
2. 若缺少 authenticated trajectory sketch records, 不应把未认证 trajectory logging 写成高置信证据。
3. 若缺少 fixed-FPR 外部 baseline 对比, 不应宣称已满足最终投稿结果闭环。
4. 若 placeholder 字段参与某一结论, 该结论只能作为待验证项, 不能作为 supported claim。

## 4. 与分阶段构建文档的对应关系

```text
docs/builds/phases/protocol_governance_foundation_build.md
docs/builds/phases/synthetic_state_inference_sanity_build.md
docs/builds/phases/real_video_latent_transfer_check_build.md
docs/builds/phases/state_space_inference_formalization_build.md
docs/builds/phases/trajectory_observation_core_probe_build.md
docs/builds/phases/flow_model_adapter_preflight_build.md
docs/builds/phases/sampling_time_constraint_probe_build.md
docs/builds/phases/small_scale_claim_pilot_gate_build.md
docs/builds/phases/generative_video_model_probe_build.md
docs/builds/phases/replay_and_authenticated_sketch_gate_build.md
docs/builds/phases/submission_package_freeze_build.md
```
