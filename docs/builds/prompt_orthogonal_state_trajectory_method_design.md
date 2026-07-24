# Prompt-Orthogonal State-Trajectory Watermark 方法设计

## 0. 文档状态

本文档定义 SSTW 在当前 phase-code 机制失败后的候选方法路线：

```text
prompt-orthogonal state-dependent key deflection
+ continuous balanced trajectory code
+ replay innovation sequence
+ synchronous vector demodulation
```

候选中文名称为：

```text
提示条件正交的状态轨迹水印
```

候选学术表述为：

```text
Training-free prompt-nuisance-orthogonal state-trajectory watermarking
for pretrained video Flow Matching models.
```

本文档属于 `gpu_mechanism_smoke_ready_not_executed` 阶段。2026-07-24 已完成
P0--P9 的本地实现：generation scheduler hook、共享 Wan replay candidate evaluator、
冻结 experiment decision、双 source fail-closed 校验以及不修改 Notebook 的 Colab
白名单 handler。全量本地门禁通过前、真实 Colab/GPU 运行前，它仍不是 GPU 结果或论文
证据。本文档不得
被解释为：

- 当前方法已经通过机制验证；
- 已允许进入攻击、fixed-FPR、外部 baseline 或 paper profile；
- 已证明 state-space trajectory watermark 在 Wan 上可行；
- 已授权修改 Colab Notebook 或启动 GPU；
- 已获得正式 calibration、攻击或 paper evidence 的执行授权。

本轮 records、thresholds 与 claim downgrade 已登记；真实 GPU 结果仍必须按冻结门禁
判定，禁止依据观察值回调阈值。

## 1. 调整依据

最新 temporal-code isolation 诊断满足以下完整性条件：

- replay runtime 无 failure；
- 4 个 signed 视频全部完成固定 20-step replay；
- 36 条 summary、32 条 owner/wrong pair、4 条 identity 完整；
- 空间 key 固定为 owner；
- wrong temporal codes 互异且满足预声明相关性上限；
- 原始 records 可独立重算出相同 decision。

但冻结门禁得到：

| 诊断量 | 观察值 | 冻结要求 | 结论 |
|---|---:|---:|---|
| owner-over-wrong pair fraction | 0.5625 | 不低于 0.75 | 失败 |
| owner top-1 identity fraction | 0.25 | 不低于 0.75 | 失败 |
| prompt pair-fraction gap | 0.75 | 不高于 0.25 | 失败 |
| minimum replay reliability | 0.6642 | 不低于 0.05 | 通过 |

Prompt 003 获得 15/16 pair wins，而 Prompt 004 仅获得 3/16。该结果不能证明抽象的
state-space watermark 不可行，但足以否定以下当前组合：

```text
prompt-bound binary phase code
+ fixed additive tubelet direction
+ scalar S_path_inv identity gate
```

因此下一轮不得通过增大 `lambda`、扩大候选池、增加 phase grid、降低门槛或直接扩样
修补当前实例。

## 2. 相关研究与可借鉴边界

### 2.1 PAI：key-conditioned trajectory deflection

PAI 在初始噪声嵌入之外，于早期 denoising steps 执行 key-conditioned deflection，
并在验证时使用与 key 对应的逆过程。它同时指出，一维标量难以表达不同方向的 latent
bias，因此使用低维 bias representation 与统计距离。

SSTW 可借鉴：

- key-conditioned deflection 必须在生成与验证中具有同源逆语义；
- 检测应保留向量方向，而不是过早压缩成单标量；
- 正确 key 与错误 key 的差异应体现为结构化 innovation，而不只是 endpoint 大小。

SSTW 不直接采用：

- 以 initial-noise reconstruction bias 作为主要证据；
- 图像 DDIM 专用的乘性 deflection；
- 以 timestamp metadata 作为核心识别条件。

来源：
Qingyu Liu et al.,
[Attack-Resistant Watermarking for AIGC Image Forensics via Diffusion-based Semantic Deflection](https://arxiv.org/abs/2601.06639),
2026。

### 2.2 Dynamics-Level Flow Matching Watermark

该工作把 flow-matching velocity field 看作连续通信信道，通过零积分时间载波、随机
codeword 和 synchronous demodulation 恢复消息。

SSTW 可借鉴：

- 连续时间、零积分载波；
- codeword 归一化与 wrong-key chance-level 对照；
- synchronous demodulation；
- 以连续 schedule 函数跨不同离散网格重建。

SSTW 不直接采用：

- 训练期修改 flow-matching target；
- 假设可对模型执行大量独立 black-box velocity queries；
- 将 MNIST/CIFAR 的结果外推到预训练 Wan 视频模型。

来源：
Shuchan Wang,
[Dynamics-Level Watermarking of Flow Matching Models with Random Codes](https://arxiv.org/abs/2605.16239),
2026。该工作目前只作为近期设计参考，不是 SSTW 可行性的外部证明。

### 2.3 Initial-noise 与视频时序水印

Tree-Ring、VideoShield、VideoMark 和 SIGMark 主要通过 watermarked initial noise、
inversion、frame-wise code 或 temporal alignment 完成检测。RingID 进一步指出，
multi-key identification 不能依赖 watermark 引起的公共 distribution shift。

SSTW 可借鉴：

- codebook 的多 key 分离必须显式验证；
- 视频帧删除、插入和重排需要独立的 sequence alignment；
- initial-noise watermark 可作为未来 external/internal baseline。

SSTW 不直接采用：

- 将 initial-noise template 作为主方法；
- 把视频 frame-time alignment 当作 flow-time trajectory synchronization；
- 通过公共 distribution shift 代替 key identity。

来源：

- [Tree-Ring Watermarks](https://arxiv.org/abs/2305.20030)
- [RingID](https://arxiv.org/abs/2404.14055)
- [VideoShield](https://openreview.net/forum?id=uzz3qAYy0D)
- [VideoMark](https://arxiv.org/abs/2504.16359)
- [SIGMark](https://arxiv.org/abs/2603.02882)

## 3. 核心研究问题

给定预训练 video Flow Matching 模型、prompt \(p\)、master key \(K\) 和 latent state
\(z_t\)，能否在不训练模型的前提下，引入低能量、prompt-independent 的状态转移
innovation，使输出视频经 inversion/replay 后，正确 key 能通过完整 innovation
sequence 与错误 key 分离？

候选方法的关键约束是：

1. **Prompt 不定义码字。** Prompt 只参与 clean velocity 预测与 nuisance 消除。
2. **载体依赖当前状态。** Key 定义状态空间算子，不定义固定加性空间模板。
3. **时间码是连续函数。** Generation 与 replay 在不同 grid 上重建同一函数。
4. **检测保留向量结构。** 单标量只能是最终校准统计量，不能是唯一事实来源。
5. **Primary evidence 来自 trajectory innovation。** Initial noise、endpoint 和服务端
   sketch 只能是 control 或辅助证据。

## 4. 两个时间轴

视频生成包含两个不同的时间轴：

| 时间轴 | 符号 | 含义 | 当前候选路线 |
|---|---|---|---|
| Flow/diffusion time | \(t\) | sampler 从 noise 到 video latent 的状态演化 | 主方法 |
| Video frame time | \(\tau\) | 输出视频内部帧序列 | 首轮固定聚合，后续鲁棒扩展 |

首轮机制 smoke 只验证 flow-time trajectory code。不得在同一次实验中加入 frame
deletion alignment、segment ordering 或 optical flow，以免无法归因。

## 5. Prompt-independent key derivation

Master key 通过稳定的方法域与 latent layout 域分离派生两个子 key：

\[
K_{\mathrm{state}}
=
\operatorname{KDF}(K,\texttt{state_operator},D),
\]

\[
K_{\mathrm{time}}
=
\operatorname{KDF}(K,\texttt{trajectory_code},D),
\]

其中 \(D\) 只描述方法语义、latent layout、operator schema 和 code schema。模型 ID、
revision、sampler family 与实际 schedule 必须进入 provenance 和兼容性检查，但不得
改变 watermark identity。KDF 输入禁止包含：

- prompt text；
- prompt digest；
- seed；
- 精确 model revision/hash；
- sampler implementation version；
- 离散 step index；
- generation/replay 的具体 step count；
- 从当前实验结果选择的 strength、window 或 carrier。

Prompt 必须作为 nuisance condition 与 key identity 分离。

## 6. 状态依赖 key operator

由 \(K_{\mathrm{state}}\) 生成两个单位低秩向量 \(a_K,b_K\)，构造反对称算子：

\[
A_K=a_Kb_K^\top-b_Ka_K^\top.
\]

对当前状态 \(z_t\)，状态依赖的旋转切向量为：

\[
q_{K,t}=A_Kz_t.
\]

该构造具有：

\[
\langle z_t,A_Kz_t\rangle=0,
\]

因此 \(q_{K,t}\) 描述当前 latent 在 keyed two-dimensional plane 上的切向旋转，而不是
向所有样本添加相同空间模板。

实际实现不得物化完整矩阵 \(A_K\)，而应使用低秩形式：

\[
A_Kz_t=a_K\langle b_K,z_t\rangle-b_K\langle a_K,z_t\rangle.
\]

向量 \(a_K,b_K\) 的 shape、channel grouping 和 video-frame aggregation 必须在配置中
冻结；不得通过当前测试结果选择。

## 7. Prompt-nuisance orthogonalization

给定原始模型 velocity：

\[
v_t=v_\theta(z_t,t,p),
\]

定义 nuisance subspace \(\mathcal{N}_t\)。最薄实现只允许使用当前状态与当前 model
velocity 构造 key-independent nuisance basis，不允许训练 prompt classifier 或使用
test-result-dependent PCA：

\[
\mathcal{N}_t=\operatorname{span}\{z_t,v_t\}.
\]

将 keyed tangent 投影到 nuisance orthogonal complement：

\[
\bar q_{K,t}
=
\Pi^\perp_{\mathcal{N}_t}q_{K,t},
\qquad
u_{K,t}
=
\frac{\bar q_{K,t}}
{\|\bar q_{K,t}\|_2+\epsilon}.
\]

如果投影后范数不足，当前 step 必须 inactive；不得回退到未投影方向。后续若要增加
cross-attention、text-gradient 或 learned nuisance basis，必须作为独立候选方法和
消融，不得静默替换最薄实现。

## 8. Continuous balanced trajectory code

由 \(K_{\mathrm{time}}\) 选择预声明连续基函数上的单位 codeword：

\[
a_K(t)
=
\sum_{m=1}^{d_c}c_{K,m}\phi_m(t),
\qquad
\|c_K\|_2=1.
\]

首轮候选只允许固定 \(d_c\) 的低维平滑基。基函数必须满足真实 schedule 权重下的
零均值：

\[
\sum_i w_i a_K(t_i)=0,
\]

且不得通过中心化产生近零 active code。Generation grid 与 replay grid 可分别执行
加权中心化，但必须共享：

- continuous basis family；
- master codeword；
- function digest；
- model/sampler domain separation。

禁止使用单个 keyed transition threshold 决定全部符号。

## 9. Training-free embedding

候选 velocity increment 为：

\[
\delta v_{K,t}
=
\lambda(t)a_K(t)u_{K,t}.
\]

水印后的 velocity 为：

\[
v_t^{\mathrm{wm}}=v_t+\delta v_{K,t}.
\]

必须保留三类 guard：

1. per-step relative norm guard；
2. cumulative scheduler-weighted energy guard；
3. state-tangent/nuisance-orthogonality guard。

三类 guard 均以送入 scheduler 的实际 FP32 control 与原始 velocity 的差值为准。
edge-of-window 的微小 analytic delta 若因 `base + delta` 舍入略超原预算，只允许
在冻结预算与方向阈值内做有界、确定性的 actual-delta backoff，并选择搜索到的
最大可行非零控制；搜索无解时必须带安全标量诊断 fail-closed，不得把 active
step 静默改为 no-op，也不得放宽阈值。

候选路线明确移除：

- 独立 DC allocation；
- endpoint minimum-energy controller；
- endpoint response 作为当前机制 smoke 的 primary gate；
- 正负码之外的第四强度或 adaptive strength grid。

Endpoint drift 只作为质量和机制诊断，不参与首轮 identity positive。

## 10. Replay innovation sequence

对输出视频执行一次 key-independent inversion/replay，得到固定状态序列
\(\hat z_0,\ldots,\hat z_N\)。对每一步计算 key-independent clean innovation：

\[
r_i
=
\frac{\hat z_{i+1}-\hat z_i}{\Delta\sigma_i}
-v_\theta(\hat z_i,t_i,p).
\]

正确 key 和错误 key 共享同一 replay states、同一 model velocity 和同一 reliability
weights。候选 key 只改变：

- keyed state operator；
- continuous trajectory code；
- matched demodulation direction。

这条约束用于防止 candidate-specific replay 路径制造分离。

## 11. Synchronous vector demodulation

对候选 key 计算逐步投影：

\[
y_{K,i}
=
\left\langle r_i,u_{K,i}\right\rangle.
\]

对每个连续基函数保留一个通道：

\[
h_{K,m}
=
\sum_i
w_i\rho_i\phi_m(t_i)y_{K,i},
\]

其中 \(\rho_i\) 是 key-independent replay reliability。得到向量：

\[
h_K=(h_{K,1},\ldots,h_{K,d_c}).
\]

候选匹配不得直接使用原始求和。先使用 calibration-only、candidate-independent 的
位置尺度或 covariance 进行 whitening：

\[
\tilde h_K=W(h_K-\mu_0).
\]

然后以 codeword matched likelihood 或广义似然比比较候选：

\[
T_K
=
-\left\|
\tilde h_K-\alpha c_K
\right\|_2^2,
\]

其中 \(\alpha\) 必须由 calibration 或预声明解析估计得到，不得在 test identity 上为
每个 key 自适应选择。

首轮 smoke 必须保存 \(y_{K,i}\)、\(h_K\)、whitening 状态和候选排名。标量 \(T_K\)
只能用于候选排序，不能替代向量 records。

## 12. 检测判定与证据层级

候选路线分三层判定：

1. **Construction readiness**：数学、shape、grid、budget 与 candidate-independence
   不变量通过。
2. **Mechanism smoke**：owner/wrong identity、prompt gap、clean control 和质量通过。
3. **Formal detection**：独立 calibration/test、fixed-FPR、攻击和 baseline 通过。

当前只允许设计并实现前两层。即使 mechanism smoke 通过，也只能允许设计 formal
calibration；不得直接声称水印有效或进入论文主结果。

## 13. 最小 GPU 验证协议

首轮 GPU 运行必须在实现和本地审计通过后单独授权。协议冻结为：

```text
2 prompts
x 2 seeds
x (prompt_orthogonal_state_trajectory / clean)
= at most 8 generated videos
```

若存在模型、prompt、seed、frame count、resolution、scheduler 和 revision 全部一致的
hash-validated clean videos，可复用 clean；否则才生成 4 个 clean。

每个 watermarked video：

- 一次 20-step key-independent replay；
- owner 对 8 个预声明 wrong keys；
- 不运行攻击；
- 不运行 strength grid；
- 不运行 state posterior；
- 不运行 fixed-FPR；
- 不运行 external baseline；
- 不修改 Notebook，只增加白名单 workflow handler。

最小门禁至少包含：

| Gate | 冻结方向 |
|---|---|
| owner-over-wrong pair fraction | 不低于 0.75 |
| owner top-1 identity fraction | 不低于 3/4 |
| prompt pair-fraction gap | 不高于 0.25 |
| replay reliability | 不低于现有 smoke 下限 |
| clean owner-like response | 不得与 watermarked owner 同级 |
| quality/energy guards | 全部通过 |

上述数值已在
`configs/protocol/sstw_prompt_orthogonal_state_trajectory_smoke.json`
及 mutation rejection tests 中冻结；本文档不代替字段注册和协议配置。额外冻结
replay reliability 不低于0.05、watermarked owner 严格胜过同 identity clean owner
的比例不低于0.75、clean owner top-1 比例不高于0.5。

## 14. 必要消融

机制通过后，以下消融才允许设计：

1. without prompt-nuisance projection；
2. fixed additive direction 替代 state-dependent operator；
3. scalar path integral 替代 vector demodulation；
4. prompt-bound code derivation；
5. non-balanced temporal carrier；
6. owner state operator + wrong temporal code；
7. wrong state operator + owner temporal code；
8. endpoint-only / initial-noise baseline。

首轮 smoke 不得一次性运行全部消融。

## 15. 可证伪条件

满足任一条件时，必须停止当前候选实例：

- Prompt 003/004 再次出现系统性方向翻转；
- owner top-1 低于 3/4；
- prompt gap 高于 0.25；
- 分离主要由公共 norm、endpoint drift 或 distribution shift 产生；
- correct/wrong separation 在去除 prompt digest 后消失；
- vector detector 只能通过 test-time whitening、carrier selection 或 strength tuning
  才能通过；
- state-dependent operator 被模型完全吸收，innovation 不可观察；
- clean videos 对 owner key 产生同等级 matched response；
- quality 或 energy guard 需要放宽才能获得分离。

若最薄 state-dependent operator 失败，SSTW 应重新评估 training-free trajectory
watermark 主线，而不是继续增加 posterior、攻击或治理层。

## 16. 候选创新边界

如果后续证据通过，SSTW 的候选创新应限定为：

```text
面向预训练视频 Flow Matching 模型的 training-free 状态依赖轨迹水印；
通过 prompt-nuisance-orthogonal key operator 在 sampling dynamics 中形成
低能量 innovation，并通过输出视频反演得到的完整 innovation vector 执行
synchronous key demodulation。
```

不得声称：

- 首个 trajectory watermark；
- 首个 flow-matching watermark；
- 理论上不可移除；
- 完全黑盒检测；
- 无需模型即可验证；
- 已优于 PAI、Tree-Ring、VideoShield、VideoMark 或 SIGMark；
- 对 frame attacks 鲁棒，除非后续独立实现 video-time alignment 并验证。

## 17. 与现有实现的迁移关系

| 当前实现语义 | 候选路线处理 |
|---|---|
| `flow_tubelet_key_code` 固定加性方向 | 不作为新 primary carrier；保留历史 control |
| prompt digest 参与 carrier context | 从 key/code derivation 移除 |
| binary/multisegment phase code | 替换为连续低维 balanced code |
| `S_path_inv` 标量 | 降级为历史 control/diagnostic |
| endpoint/DC controller | 首轮候选禁用 |
| key-independent replay states | 保留 |
| replay reliability weighting | 保留，但必须 candidate-independent |
| candidate spatial/temporal decoupling | 保留为机制消融 |
| Colab 薄入口与本地后打包 | 保持不变 |

具体算法接口和不变量定义在
`docs/builds/prompt_orthogonal_state_trajectory_algorithm_primitives.md`。
