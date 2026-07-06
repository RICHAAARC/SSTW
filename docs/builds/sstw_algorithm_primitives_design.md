# SSTW 项目算法原语：状态空间同步 Flow Matching 轨迹水印

## 0. 文档定位

本文档用于学术性说明 SSTW 项目的算法是什么、由哪些算法原语组成、体系创新性在哪里, 以及每个原语应通过哪些实验被验证。本文档可以独立阅读, 不要求读者先阅读 `sstw_method_mechanism_design.md`。

SSTW 的目标不是给最终视频帧叠加一个后处理水印, 也不是用 DTW 或帧匹配恢复时间轴后再做普通检测。SSTW 的核心算法是:

```text
在 Flow Matching 视频生成采样过程中, 用密钥条件弱约束改变速度场轨迹, 再用路径积分证据、终点证据和 replay 不确定性感知状态后验共同完成 fixed low-FPR 检测。
```

学术表述为:

```text
Key-conditioned velocity-field watermarking and replay-aware path-posterior inference for Flow Matching video generation trajectories.
```

## 1. 问题设定

给定文本条件 `c`、随机种子 `s`、生成模型 `G_theta` 和密钥 `K`, Flow Matching 视频生成可抽象为:

```text
dz_t / dt = v_theta(z_t, t, c)
```

离散采样时, latent 状态按时间网格更新:

```text
z_{i+1} = z_i + Delta_t_i * v_theta(z_i, t_i, c) + numerical_error_i
```

SSTW 在采样阶段引入低能量密钥条件扰动, 使水印证据同时出现在四个层面:

```text
velocity evidence
path-integral evidence
endpoint evidence
replay posterior evidence
```

最终检测不是单一分数阈值, 而是受 admissibility 与 calibration negative 约束的假设检验。

## 2. 算法原语总览

| 原语 | 名称 | 作用 | 创新性来源 | 必须验证 |
|---|---|---|---|---|
| P1 | Flow Tubelet Key Code | 生成 tubelet、payload、flow phase 的统一密钥码 | 把 payload 与 flow phase 绑定到同一轨迹编码 | wrong-key / shuffled-key 分离 |
| P2 | Velocity-Field Weak Watermark Constraint | 在采样速度场中加入弱水印约束 | 水印进入生成动力学而非后处理 | velocity alignment gain > 0 |
| P3 | Endpoint-Aware Minimum-Energy Control | 控制扰动能量并保持 endpoint 可读 | 将路径水印与终点 payload 统一 | quality 与 endpoint evidence 同时通过 |
| P4 | Time-Reparameterization-Invariant Path Observation | 构造对采样步数和 time grid 更稳的路径证据 | 避免退化为固定 step-index 对齐 | wrong grid / wrong sampler 退化可解释 |
| P5 | Replay-Uncertainty-Aware Flow-State Inference | 用状态空间后验融合路径、终点、replay 证据 | 显式建模 replay 不确定性 | 优于 generic SSM / GRU / Transformer |
| P6 | Flow-State Evidence Admissibility | 限制状态搜索与证据准入 | 抑制 negative false positive tail | without-admissibility FPR 上升 |
| P7 | Fixed Low-FPR Calibration | 用 calibration negative 冻结阈值 | 把轨迹搜索纳入低误报协议 | held-out FPR 受控 |
| P8 | Authenticated Trajectory Sketch | 认证 owner-side 轨迹摘要 | 防止日志审计被伪造 | tampered sketch 被拒绝 |
| P9 | Baseline-Separated Evaluation | 与外部视频水印和显式同步 baseline 分离 | 证明增益来自 SSTW 机制 | beats modern baselines |
| P10 | Governed Artifact Rebuild | records 到 tables / figures / reports 自动重建 | 防止手工 claim 与结果泄漏 | claim audit pass |

## 3. P1: Flow Tubelet Key Code

该原语把视频 latent 切分为时空 tubelet, 并为每个 tubelet 生成密钥条件方向、payload bit 和 flow phase code。

输入:

```text
key K
tubelet index g
normalized flow phase phi(t_i)
prompt digest
sampler signature 或 sampler phase abstraction
```

输出:

```text
u_{i,g}(K): key-conditioned direction
b_g: payload bit
pi_{i,g}(K): flow phase code
c_{i,g}(K): joint code
```

创新点在于: 同一个 tubelet code 同时服务于速度场约束、路径积分观测、终点响应和状态后验推断, 而不是分别设计互不相干的 payload code 和 temporal sync code。

## 4. P2: Velocity-Field Weak Watermark Constraint

该原语在 Flow Matching 采样过程中加入弱速度场增量:

```text
delta_v_{i,g} = lambda(t_i) * a_{i,g} * c_{i,g}(K) * P_g u_{i,g}(K)
```

其中 `lambda(t_i)` 是时间调度, `a_{i,g}` 是内容、运动、语义安全和可控性强度。

该原语与普通后处理水印的区别是: 水印扰动作用于生成轨迹中的速度或 latent displacement proxy, 因而必须记录 callback trajectory、time grid、sampler signature 和 velocity proxy。

## 5. P3: Endpoint-Aware Minimum-Energy Control

仅在路径中加入扰动不足以支撑可检测水印, 因为轨迹证据可能在最终视频中消失。该原语要求扰动在能量预算内尽量保持 endpoint evidence 可读。

工程实现可以从近似方案开始:

```text
finite_difference_controllability_gate
latent_response_gain_proxy
endpoint_consistency_score
quality_guard
semantic_projection_status
```

该原语的主张不是严格最优控制, 而是“endpoint-aware minimum-energy approximation”。

## 6. P4: Time-Reparameterization-Invariant Path Observation

该原语把相邻状态增量投影到密钥方向, 并进行步长归一化:

```text
I_inv_g(K) = sum_i beta_i * c_{i,g}(K) * <z_{i+1,g}-z_{i,g}, u_{i,g}(K)> / (||z_{i+1,g}-z_{i,g}|| + eps)
```

设计目的:

1. 减弱对固定 step index 的依赖。
2. 支持不同 scheduler、step count、time grid 的比较。
3. 让 wrong sampler replay 出现可解释退化, 而不是完全伪造正例。

## 7. P5: Replay-Uncertainty-Aware Flow-State Inference

攻击后视频无法直接恢复原始生成轨迹, 因此 replay 或 inversion 得到的是带不确定性的近似轨迹。该原语用状态空间模型估计水印状态:

```text
h_t = [phase, endpoint, confidence, temporal_disturbance, path_consistency, velocity_consistency, replay_reliability, time_grid_reliability]
```

观测向量包含:

```text
endpoint response
velocity response
path response
path-endpoint consistency
quality / coverage
replay uncertainty
key-conditioned time embedding
```

该原语必须与 generic SSM、GRU、Transformer temporal aggregator 和 key-agnostic SSM 对比。

## 8. P6: Flow-State Evidence Admissibility

轨迹 replay 与状态搜索会扩大假设空间, 如果不加约束, negative sample 可能产生伪轨迹。因此该原语定义证据准入门:

```text
endpoint evidence pass
path evidence pass
path-endpoint consistency pass
posterior confidence pass
coverage pass
state entropy pass
negative tail pass
replay reliability pass
time-grid reliability pass
```

任何单一证据层都不能绕过 admissibility 直接触发 positive。

## 9. P7: Fixed Low-FPR Calibration

该原语使用 calibration negative 冻结阈值, 并在 held-out test split 中报告 FPR。full_paper 目标必须达到:

```text
TPR@FPR=0.01
TPR@FPR=0.001
```

`FPR=0.001` 要求大规模 negative event。若 negative event 数不足, 只能报告 pilot_paper 或 validation 结果; 其中 pilot_paper 可以支撑小样本论文级 `TPR@FPR=0.01` 主张, 但不能支撑 full_paper 规模 claim。

## 10. P8: Authenticated Trajectory Sketch

Owner-side trajectory audit 不能退化为未认证日志。该原语要求服务端保存压缩轨迹摘要并签名或摘要绑定:

```text
Hash(key_id, trajectory_sketch, prompt, seed, model_signature, sampler_signature)
Sign_server(hash)
```

该原语不是密码学主贡献, 但它是部署可信度和 owner-side evidence 的必要协议组件。

### 10.1 replay/sketch gate 的原语组合

`replay/sketch gate` 不是单独替代 P5 或 P8 的算法原语, 而是 Claim-3 的原语组合门禁。它要求 P5 的 replay-uncertainty-aware posterior、P6 的 admissibility、P7 的 fixed low-FPR calibration 和 P8 的 authenticated trajectory sketch 同时形成 governed evidence。

该 gate 的算法职责是:

1. 验证 authenticated sketch 没有被替换或伪造。
2. 验证 replay uncertainty 已参与 posterior weighting, 而不是只作为报告指标。
3. 验证 wrong sampler、wrong prompt 和 wrong time grid replay 不能伪造正确轨迹。
4. 验证 replay negative 与 sampler-mismatch negative 的 FPR tail 受控。
5. 在证据不足时触发 Claim-3 downgrade, 防止 unsupported robust replay verification claim。

因此, 项目可以短期使用 `Claim-3 downgrade gate` 推进 validation_scale, 但最终若要把 Claim-3 作为强创新性贡献, 必须实现并通过 `replay/sketch gate`。

## 11. P9: Baseline-Separated Evaluation

SSTW 的实验必须同时对比:

```text
modern_video_watermark_baseline
explicit_synchronization_baseline
internal_mechanism_baseline
state_model_baseline
flow_specific_control
```

外部 baseline 不能只包含 DTW 或帧匹配, 因为它们只能证明 SSTW 不是显式同步, 不能证明 SSTW 优于现代视频水印方法。

## 12. P10: Governed Artifact Rebuild

所有论文主张必须从 governed records 重建。该原语属于工程治理原语, 但对顶刊顶会论文同样重要, 因为它能降低手工表格、阈值泄漏和 unsupported claim 风险。

必须支持:

```text
records -> thresholds
records + thresholds -> tables
records + manifests -> figures
records + manifests -> reports
reports -> claim audit
```

## 13. 体系创新性

SSTW 的体系创新性不在于单独引入状态空间模型, 也不在于单独做视频水印, 而在于以下组合:

1. 将视频水印嵌入对象从最终帧或最终 latent 扩展到 Flow Matching 生成轨迹。
2. 用统一的 flow tubelet key code 同时驱动 payload、velocity constraint、path evidence 和 posterior inference。
3. 用时间重参数化不变路径积分避免固定 step-index 同步。
4. 用 replay uncertainty 显式处理攻击后视频的轨迹恢复误差。
5. 用 admissibility 和 fixed low-FPR calibration 控制状态搜索带来的 false positive tail。
6. 用 authenticated trajectory sketch 区分可信 owner-side audit 与普通日志。
7. 用 governed artifact rebuild 将实验结果与论文 claims 绑定。

## 14. 与已有路线的边界

| 路线 | 与 SSTW 的关系 | 不能替代的部分 |
|---|---|---|
| 后处理视频水印 | 外部 baseline | 不能证明水印进入生成轨迹 |
| 图像水印逐帧扩展 | baseline 或弱对照 | 缺少 flow path evidence |
| DTW / frame matching | 显式同步 control | 缺少 key-conditioned state posterior |
| generic SSM / GRU / Transformer | 状态模型 baseline | 缺少 Flow trajectory 语义和 key condition |
| 服务端日志 | 部署组件 | 缺少 watermark constraint 与 admissibility |
| endpoint-only latent watermark | 内部消融 | 缺少 path marginal gain |

## 15. 论文 claim 映射

| claim | 必须原语 | 必须 records | 失败时降级 |
|---|---|---|---|
| Claim-1: velocity-field watermarking | P1, P2, P3, P7 | velocity alignment, endpoint evidence, quality records | 降级为 endpoint-aware latent watermark |
| Claim-2: path-posterior inference | P4, P5, P6, P7 | path gain, state posterior, negative tail records | 降级为 state-space synchronized watermark |
| Claim-3: robust replay verification | P5, P6, P7, P8 | replay uncertainty, wrong sampler, wrong prompt, sketch verification, replay/sketch gate decision | 限定为 owner-side audit 或附录 |
| Claim-4: top-tier empirical robustness | P7, P9, P10 | baseline, ablation, attack, full_paper records | 不进入 full_paper claim |

## 16. 实验充分性矩阵

每个算法原语都必须映射到至少一个正向证据、一个反事实 control 和一个失败诊断项。该矩阵用于防止论文被评价为“只有主表分数, 缺少机制证明”。

| 原语 | 正向证据 | 反事实 control | 失败诊断 |
|---|---|---|---|
| P1 | correct-key positive score 分离 | wrong-key、shuffled-key、key-agnostic | key direction collision |
| P2 | velocity / latent displacement alignment gain | without-velocity、wrong-key constraint | gain <= 0 或 quality collapse |
| P3 | endpoint evidence 与质量同时通过 | random controllability、without endpoint control | endpoint-path decoupling |
| P4 | path marginal gain at fixed-FPR > 0 | raw path、time-shuffled path、wrong time grid | path 与 endpoint 高冗余 |
| P5 | state posterior 优于 generic SSM | GRU、Transformer、generic SSM、key-agnostic SSM | negative tail inflation |
| P6 | admissibility 降低 FPR tail | without-admissibility | TPR 大幅下降且 FPR 不降 |
| P7 | held-out FPR 受控 | test-time threshold update blocked | calibration leakage |
| P8 | tampered sketch 被拒绝 | unsigned logging、sketch replacement | sketch 无法认证 |
| P9 | full method 优于现代 baseline | external baseline records | baseline not runnable |
| P10 | tables / figures 可重建 | manual table blocked | unsupported claim |

## 17. 拒稿风险与算法反驳路径

顶会审稿中常见质疑应由算法原语与实验共同回答:

| 可能质疑 | 算法反驳路径 | 实验反驳路径 |
|---|---|---|
| 只是后处理视频水印 | P2 证明 sampling-time 约束进入轨迹 | velocity alignment 与 callback trace records |
| 只是 endpoint latent watermark | P4 证明 path evidence 有独立增益 | endpoint-only vs full method at fixed-FPR |
| 只是普通状态空间模型 | P5 使用 key-conditioned flow-state posterior | generic SSM / GRU / Transformer baseline |
| replay 会制造伪轨迹 | P6 与 P7 控制 negative tail | replay negative、wrong sampler、wrong prompt |
| 低 FPR 不可信 | P7 固定 calibration negative | held-out FPR=0.001 + confidence interval |
| 结果不可复现 | P10 强制 artifact rebuild | manifests、records、rebuild command |

如果某个质疑没有对应的实验反驳路径, 则该 claim 不得进入 full_paper 主贡献。

## 18. 原语到 full_paper package 的映射

full_paper package 不应只包含最终主表。它必须让读者从 package 中追溯每个原语的证据链:

```text
P1 -> key_condition_records.jsonl
P2 -> velocity_constraint_records.jsonl
P3 -> endpoint_quality_control_records.jsonl
P4 -> trajectory_path_records.jsonl
P5 -> state_posterior_records.jsonl
P6 -> admissibility_audit_records.jsonl
P7 -> thresholds.jsonl
P8 -> trajectory_sketch_verification_records.jsonl
P5+P6+P7+P8 -> replay_and_sketch_gate_decision.json
P9 -> baseline_scores.jsonl
P10 -> full_paper_package_manifest.json
```

缺少任一必要记录时, 对应原语只能作为设计说明, 不能作为 supported claim。

## 19. 原语可证伪条件

为了避免把算法原语写成不可检验的设计叙述, 每个原语必须有明确可证伪条件。若触发可证伪条件, 论文应降级对应 claim。

| 原语 | 可证伪条件 | 必须降级的 claim |
|---|---|---|
| P1 | wrong-key、shuffled-key 与 correct-key 分布无法分离 | key-conditioned watermark claim |
| P2 | keyed velocity / displacement gain 不显著或方向不稳定 | Flow velocity watermarking claim |
| P3 | endpoint evidence 与 path evidence 长期解耦 | endpoint-aware flow control claim |
| P4 | path score 与 endpoint score 高冗余且无 fixed-FPR 增益 | trajectory path observation claim |
| P5 | generic SSM 或普通 temporal aggregator 与 full method 等价 | state-space posterior innovation claim |
| P6 | admissibility 不降低 negative tail 或显著牺牲 TPR | admissibility claim |
| P7 | threshold 需要 test split 调整才能控制 FPR | low-FPR calibrated detection claim |
| P8 | trajectory sketch 可被替换或 replay 而不被拒绝 | authenticated audit claim |
| P9 | 现代外部 baseline 未运行且无合理协议解释 | top-tier comparison claim |
| P10 | tables、figures 或 reports 无法从 records 重建 | governed evidence claim |

这些可证伪条件属于通用科研工程写法, 不是 SSTW 特有机制。SSTW 的项目特定创新在于将这些可证伪条件绑定到 Flow Matching 轨迹水印的算法原语与 governed artifacts。

## 20. 原语级实验打包要求

full_paper 结果包中不应只保留主表。每个原语都应能被独立审计:

```text
primitive_id
primitive_name
positive_evidence_record
counterfactual_control_record
ablation_table
failure_diagnostic_report
claim_support_status
claim_downgrade_rule
```

若某个原语只有文字说明, 没有 positive evidence 和 counterfactual control, 则该原语只能进入方法描述, 不能作为论文贡献之一。
