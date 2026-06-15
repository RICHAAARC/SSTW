# SSTW 论文叙事注意事项：独立顶会投稿与实质性非重叠边界

## 0. 文档定位

本文档用于指导 SSTW 的论文叙事、方法命名、baseline 命名、贡献表述和实验组织。目标是使 SSTW 成为一篇独立的顶会论文, 而不是任何真实视频显式同步水印论文的扩展版。

SSTW 的核心叙事必须围绕:

\[
\boxed{
\text{video generative watermarking as key-conditioned state-space inference over generative trajectories}
}
\]

中文表述为:

> 将视频生成水印检测建模为生成轨迹上的密钥条件状态空间推断。

---

## 1. 总体叙事原则

### 1.1 必须强调的问题定义

SSTW 的核心不是“更强的同步搜索”, 而是:

> 生成式视频水印中的时间失同步, 本质上是攻击观测与生成轨迹共同约束下的密钥条件隐状态估计问题。

推荐英文表述:

> We formulate temporal desynchronization in video generative watermarking as key-conditioned hidden-state inference over latent tubelet and generative trajectory observations.

推荐中文表述:

> 本文将视频生成水印中的时间失同步问题形式化为 latent tubelet 与生成轨迹观测上的密钥条件隐状态估计问题。

### 1.2 必须避免的叙事

禁止或尽量避免:

```text
更强的显式同步
在显式同步基础上
对已有显式同步机制进行升级
把 SSM 加到已有显式同步方法中
继承上一工作中的 payload safety
前作
原方法
上一阶段方法
our prior method
original method
extension of prior sync method
```

这些表述会使 SSTW 显得像另一篇工作的增量扩展, 从而增加实质性重叠风险。

### 1.3 推荐命名

方法标题候选:

```text
Key-Conditioned State-Space Inference for Generative Video Watermarking
State-Space Tubelet Trajectory Watermarking
Generative Trajectory State Inference for Robust Video Watermarking
```

核心模块命名:

```text
Key-conditioned tubelet code
Key-conditioned state-space inference
Generative trajectory observation
Key-state evidence admissibility
Fixed-FPR calibrated detector
Sampling-time weak constraint
```

baseline 命名:

```text
Frame-PRC
Tubelet-only
Explicit temporal alignment
Generic temporal pooling
Conv1D aggregator
GRU aggregator
Transformer aggregator
Generic SSM
Key-agnostic SSM
Classical temporal registration
```

---

## 2. 实质性非重叠规则

### 2.1 叙事独立还不够

不能只通过“不提另一篇论文”来规避重叠。顶会审稿关注的是实质内容, 包括方法、主表、主图、实验协议、结论和代码结构。

因此 SSTW 必须做到:

```text
主贡献不同
主实验不同
主表结构不同
主图机制不同
主结论不同
主代码路径不同
```

### 2.2 不得作为主线的内容

以下内容可以作为 baseline 或 sanity check, 但不得成为 SSTW 主线:

```text
真实视频 VAE latent 下的显式同步鲁棒性
frame_prc / tubelet_only / 显式同步三方法主表
H.264 / H.265 真实视频压缩鲁棒性作为唯一主实验
外部视频水印 baseline 比较作为唯一贡献
payload safety gate 的独立主张
```

### 2.3 必须作为主线的内容

SSTW 的摘要、方法图、主表和结论必须围绕:

```text
key-conditioned hidden-state inference
generative trajectory observation
state posterior smoothing
key-state admissibility
trajectory controls
cross prompt / seed / model generalization
```

### 2.4 显式时间对齐只能是反事实 baseline

推荐写法:

> To assess whether hidden-state inference is necessary, we include an explicit temporal alignment baseline under the same fixed-FPR protocol.

中文写法:

> 为检验隐状态推断是否必要, 本文设置显式时间对齐 baseline, 并在相同 fixed-FPR 协议下进行比较。

不推荐写法:

```text
我们将前一显式同步方法作为 baseline。
本文在显式同步方法基础上进一步引入状态空间。
本文继承上一工作中的 aligned payload safety。
本文是已有 tubelet sync 方法的扩展。
```

### 2.5 主表结构要求

SSTW 主表建议按以下结构组织:

```text
Group 1: weak structural baselines
  Frame-PRC
  Tubelet-only
Group 2: temporal alignment baselines
  Explicit temporal alignment
  Classical temporal registration
Group 3: generic temporal model baselines
  Generic temporal pooling
  Conv1D
  GRU
  Transformer
  Generic SSM
  Key-agnostic SSM
Group 4: proposed state inference family
  Key-conditioned state-space inference
  Key-conditioned state-space inference + trajectory observation
  Keyed state trajectory constraint, if validated
```

注意: Group 4 才是论文中心。

---

## 3. 如何避免“工程拼凑”质疑

### 3.1 统一编码对象

论文必须反复强调所有证据都来自同一个 key-conditioned tubelet code:

\[
c_g(K)=c^{\mathrm{payload}}_g(K)\cdot c^{\mathrm{sync}}_g(K)。
\]

payload evidence、temporal evidence 和 trajectory evidence 不是三个无关分数, 而是同一编码结构在不同观测空间中的投影。

推荐表述:

> All evidence terms are observations of the same key-conditioned tubelet code, rather than independently designed detector scores.

### 3.2 统一状态变量

状态空间推断必须有可解释状态:

\[
h_t=[\tau_t,e_t,\gamma_t,\epsilon_t,\rho_t]。
\]

| 状态变量 | 解释 | 对应失效模式 |

|---|---|---|

| \(\tau_t\) | 水印相位 | temporal crop / local clip |

| \(e_t\) | 局部 payload evidence | compression / reconstruction |

| \(\gamma_t\) | 同步置信度 | spurious correlation / negative leakage |

| \(\epsilon_t\) | 时间扰动 | dropping / duplication / resampling |

| \(\rho_t\) | 轨迹一致性 | generation trajectory mismatch |

每个状态变量都必须通过消融证明不是装饰性设计。

### 3.3 Trajectory observation 必须是状态观测项

不要写成:

\[
S_{\mathrm{final}}=S_{\mathrm{tubelet}}+S_{\mathrm{state}}+S_{\mathrm{traj}}。
\]

这种写法容易被认为是工程分数融合。

推荐写成:

\[
x_t=[r^{\mathrm{pay}}_t,r^{\mathrm{sync}}_t,r^{\mathrm{traj}}_t,q_t,e_K(t)]。
\]

即 trajectory response 是状态空间推断器的观测输入, 由状态模型决定其保留、降权或忽略。

### 3.4 Key-state evidence admissibility 是算法约束

不要把 admissibility 写成简单调参门禁。它的作用是限制状态搜索和轨迹观测不能绕过 payload evidence 直接触发 positive。

必须报告:

```text
negative_state_over_threshold_count
admissibility_negative_tail_status
state_entropy_distribution
posterior_confidence_distribution
coverage_ratio_distribution
trajectory_control_suppression_status
```

### 3.5 Fixed-FPR 是方法框架的一部分

必须将 fixed-FPR 写进方法章节, 而不是只放到实验设置:

\[
\eta_\alpha=\mathrm{Quantile}_{1-\alpha}(S_{\mathrm{final}}(\mathcal{D}_{\mathrm{calib-neg}}))。
\]

并说明 calibration negative 必须包含 clean negative 和 attacked negative。

---

## 4. 贡献表述模板

### 4.1 推荐贡献表述

建议写成:

1. We formulate temporal desynchronization in generative video watermarking as key-conditioned hidden-state inference over latent tubelet and generative trajectory observations.

2. We introduce a key-conditioned tubelet code that unifies payload, temporal, and trajectory observations under a shared watermark structure.

3. We develop a key-conditioned state-space inference module that estimates watermark phase, local evidence, synchronization confidence, temporal disturbance, and trajectory consistency.

4. We propose key-state evidence admissibility to prevent state search and trajectory observation from bypassing payload evidence under fixed low-FPR calibration.

5. We show that generative trajectory observations provide non-redundant evidence beyond final decoded video frames.

6. We validate the method under fixed-FPR calibration with state-model ablations, trajectory controls, cross-prompt / seed / model generalization, and quality / motion / semantic consistency audits.

### 4.2 不推荐贡献表述

不要写成:

```text
本文把显式同步升级为状态空间同步。
本文在已有显式同步方法基础上增加 SSM。
本文融合 tubelet 分数、sync 分数和 trajectory 分数。
本文使用 Mamba / SSM 提升检测性能。
本文沿用已有 payload safety 机制。
```

这些写法会同时引发两个问题:

1. 与其他工作关联过强。

2. 被认为是工程模块堆叠, 而不是算法原语。

---

## 5. 方法章节建议结构

建议 SSTW 方法章节采用如下结构:

```text
3. Method
  3.1 Problem formulation: generative watermark state inference
  3.2 Key-conditioned tubelet code
  3.3 Key-conditioned state-space inference
  3.4 Generative trajectory observation
  3.5 Key-state evidence admissibility
  3.6 Fixed-FPR calibrated detection
  3.7 Sampling-time weak constraint, optional
```

注意:

- explicit temporal alignment 不放在 Method 主章节。

- explicit temporal alignment 只放在 Experiments 的 baselines 小节。

- sampling-time weak constraint 若未充分验证, 只能放入 appendix 或 future work。

---

## 6. 实验章节建议结构

建议实验章节采用如下结构:

```text
4. Experiments
  4.1 Experimental protocol and fixed-FPR calibration
  4.2 Generative video models and trajectory extraction
  4.3 Baselines
  4.4 Main results under temporal attacks
  4.5 State-space inference analysis
  4.6 Trajectory observation analysis
  4.7 Key condition and admissibility ablations
  4.8 Cross prompt, seed, motion and model generalization
  4.9 Quality, motion and semantic consistency
  4.10 Failure cases and limitations
```

真实视频 VAE latent transfer check 可以放入 appendix 或 supplementary, 不应成为主实验章节中心。

---

## 7. Baseline 叙事注意事项

### 7.1 内部受控 baseline

推荐命名:

```text
Frame-PRC
Tubelet-only
Explicit temporal alignment
Generic temporal pooling
Conv1D
GRU
Transformer
Generic SSM
Key-agnostic SSM
Key-conditioned state-space inference
Key-conditioned state-space inference + trajectory
Keyed state trajectory constraint
```

### 7.2 外部 baseline

可考虑:

```text
Classical temporal registration
VideoMark-style temporal matching
VideoMark
RivaGAN
VIDSTAMP
VideoShield
SIGMark
```

外部 baseline 只能在有可记录代码来源、权重来源、许可证状态、模型 digest、adapter 版本和固定 FPR 校准 records 时支撑正向 claim。

### 7.3 不要使用的 baseline 命名

```text
RelatedSyncPaper
Explicit Sync Paper
Original Sync
Previous Tubelet Sync
Our Prior Method
NumberedStage Tubelet Sync
```

---

## 8. Claim audit 设计

论文提交前必须完成 claim audit。每个主张必须绑定表格、曲线、报告或 manifest。

| Claim | 必须证据 |

|---|---|

| 状态空间推断优于显式时间对齐 | main fixed-FPR table + temporal attack breakdown |

| 不是普通时序模型替代 | Conv1D / GRU / Transformer / Generic SSM 对照 |

| key condition 必要 | key-conditioned vs key-agnostic / shuffled-key ablation |

| admissibility 必要 | admissibility ablation + negative tail audit |

| trajectory observation 有独立增益 | trajectory ablation + correlation + controls |

| trajectory evidence 不冗余 | score correlation matrix + control suppression table |

| fixed-FPR 可信 | threshold audit + attacked negative FPR |

| 生成场景成立 | generative model main table + generated video case grid |

| 泛化成立 | unseen key / prompt / seed / motion / model tables |

| sampling-time weak constraint 有效 | lambda schedule + quality robustness trade-off, if used |

若某个 claim 没有证据绑定, 必须删除、降级或移入 limitation。

---

## 9. 顶会投稿门槛

### 9.1 不满足投稿门槛的情况

以下情况不建议投稿 CVPR / ICCV / ECCV 主会:

```text
只有 synthetic latent 结果
只有 real-video VAE latent transfer 结果
主表仍然以显式同步提升为核心
trajectory observation 未通过机制审计
generative video model probe 未通过
fixed-FPR audit 未通过
state-space inference 不能优于 generic temporal models
```

### 9.2 满足最低投稿门槛的情况

```text
state_space_inference_formal_decision = PASS
trajectory_observation_mechanism_decision = PASS
generative_video_model_mechanism_decision = PASS
fixed_low_fpr_audit = PASS
claim_audit = PASS
```

### 9.3 强接收版本建议

```text
cross_generation_model_validation = PASS
trajectory_control_experiments = PASS
unseen_prompt_seed_motion_generalization = PASS
sampling_time_constraint_quality_audit = PASS
failure_case_taxonomy_ready = true
release_package_rebuildable = true
```

---

## 10. 并行投稿与重叠风险应对

### 10.1 内部审查问题

提交前必须回答:

1. SSTW 的主贡献是否依赖真实视频显式同步机制?

2. SSTW 的主表是否能在删除显式同步 baseline 后仍然成立?

3. SSTW 的主图是否围绕 state-space inference 和 generative trajectory observation?

4. SSTW 是否有生成模型轨迹实验?

5. SSTW 是否有 trajectory controls?

6. SSTW 是否有普通时序模型和 generic SSM 对照?

7. SSTW 是否有 fixed-FPR negative tail audit?

若任何问题答案为否, 则不建议投稿顶会。

### 10.2 对 concurrent work 的处理

如果存在相关并行投稿或审稿中工作, 应按目标会议政策处理。通常需要在保持匿名的前提下说明:

```text
该并行工作研究真实视频水印同步鲁棒性。
本文研究生成式视频轨迹中的密钥条件状态空间推断。
本文的主方法、主实验、主表和主结论均不同。
```

具体披露方式必须遵循目标会议作者指南。

---

## 11. 最终叙事边界

SSTW 的最终叙事边界是:

\[
\boxed{
\text{state-space inference over generative trajectories is the method; explicit alignment is only a baseline.}
}
\]

中文表述为:

\[
\boxed{
\text{生成轨迹上的状态空间推断是方法主体, 显式时间对齐只是受控对照。}
}
\]

所有方法图、公式、贡献、实验和 claim audit 都必须服务于这一边界。
