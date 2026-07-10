# SSTW 方法机制设计：状态空间同步 Flow Matching 轨迹水印

## 0. 文档定位

### 0.1 独立阅读说明

本文档可以独立阅读。读者不需要预先阅读 `sstw_algorithm_primitives_design.md`, 也可以从本文档直接理解 SSTW 的问题定义、方法机制、检测设定、baseline、消融、攻击协议和 claim audit 边界。

`sstw_algorithm_primitives_design.md` 是并列的学术化算法原语说明文档, 侧重回答“算法由哪些可复用原语组成、体系创新性在哪里、每个原语应如何被实验验证”。本文档则侧重回答“完整 SSTW 方法机制如何工作、哪些机制可以支撑论文主张、哪些失败边界必须降级”。两个文档互相引用时只能作为导航关系, 不能要求读者依赖另一个文档才能理解当前文档。

本文档定义 **SSTW** 的最终方法机制。本文方法面向基于 Flow Matching、Rectified Flow 或 velocity-field sampler 的生成式视频模型，研究对象不是最终视频帧上的显式水印，也不是对视频片段进行显式时间对齐，而是：

\[
\boxed{
\text{Key-conditioned velocity-field watermarking and path-posterior inference over Flow Matching video generation trajectories}
}
\]

中文表述为：

\[
\boxed{
\text{面向 Flow Matching 视频生成轨迹的密钥条件速度场水印与路径后验推断}
}
\]

SSTW 的核心思想是：水印信号不只存在于最终视频帧或最终 latent 中，而是以密钥条件、能量受限、语义保护的速度场弱约束进入 Flow Matching 采样动力学；水印证据沿生成路径形成可审计的时间重参数化不变路径积分统计量；检测阶段进一步通过 replay 不确定性感知的状态空间后验推断和 fixed low-FPR 校准完成判定。

因此，SSTW 不应被表述为以下机制：

```text
显式时间对齐 + 状态空间模型
最终视频帧水印 + 轨迹分数
后处理视频水印 + temporal matching
普通 latent watermark + 额外 trajectory score
Mamba / SSM 视频水印网络
trajectory logging provenance only
```

SSTW 应被表述为以下机制组合：

```text
Flow Matching velocity-field watermark constraint
+ endpoint-aware minimum-energy flow control
+ time-reparameterization-invariant path-integral evidence
+ replay-uncertainty-aware flow-state posterior inference
+ authenticated trajectory sketch for owner-side audit
+ flow-state evidence admissibility
+ fixed low-FPR calibrated detection
```

本文方法由八个算法原语组成：

```text
flow_tubelet_key_code
velocity_field_weak_watermark_constraint
endpoint_aware_minimum_energy_flow_control
time_reparameterization_invariant_path_observation
replay_uncertainty_aware_flow_state_inference
flow_state_evidence_admissibility
authenticated_trajectory_sketch
fixed_low_fpr_calibrated_detection
```

其中，`velocity_field_weak_watermark_constraint`、`time_reparameterization_invariant_path_observation` 与 `replay_uncertainty_aware_flow_state_inference` 是 SSTW 区别于普通视频水印、显式时间同步水印、通用状态空间检测器和服务端日志审计的关键机制。如果这三个机制不能在 fixed-FPR 条件下提供独立增益，SSTW 不应作为主论文核心贡献。

---

## 1. 问题定义

### 1.1 Flow Matching 视频生成过程

给定文本条件 \(c\)、随机种子 \(s\)、视频生成模型 \(G_\theta\) 和密钥 \(K\)，Flow Matching 视频生成过程可写为连续时间常微分方程：

\[
\frac{dz_t}{dt}=v_\theta(z_t,t,c),\quad t\in[0,1],
\]

其中，\(z_t\in\mathbb{R}^{F\times C\times H\times W}\) 表示视频 latent 状态，\(v_\theta\) 表示模型预测的速度场。离散采样时，设采样时间网格为：

\[
0=t_0<t_1<\cdots<t_N=1,
\]

采样更新为：

\[
z_{i+1}=z_i+\Delta t_i v_\theta(z_i,t_i,c)+\mathcal{E}_i,
\]

其中，\(\Delta t_i=t_{i+1}-t_i\)，\(\mathcal{E}_i\) 表示数值求解误差、sampler correction 或 stochastic perturbation。最终视频为：

\[
Y=D(z_N),
\]

其中，\(D\) 是视频 VAE decoder 或视频解码器。

### 1.2 水印目标

SSTW 的水印目标不是只在 \(z_N\) 中嵌入可检测模式，而是在 Flow Matching 生成过程中建立四重一致性：

\[
\boxed{
\text{velocity evidence}
\leftrightarrow
\text{path evidence}
\leftrightarrow
\text{endpoint evidence}
\leftrightarrow
\text{replay posterior evidence}
}
\]

四类证据的含义如下：

1. **Velocity evidence**：水印信号以低能量、密钥条件、时间调度的方式进入局部 tubelet 的速度场投影方向。
2. **Path evidence**：水印信号沿采样轨迹积累，形成对 sampler 网格相对稳定的路径积分统计量。
3. **Endpoint evidence**：最终视频 latent 或重建 latent 中保留与路径证据同源的 payload 投影。
4. **Replay posterior evidence**：在攻击后视频经 inversion / replay 得到的近似轨迹中，仍可恢复与密钥一致的状态后验结构。

检测目标是二元假设检验：

\[
H_0:\tilde{Y}\sim P_{\mathrm{clean}},\quad
H_1:\tilde{Y}\sim P_{\mathrm{wm}},
\]

其中，\(\tilde{Y}=\mathcal{A}(Y)\) 表示攻击后视频，\(\mathcal{A}\) 可以包含 compression、resize、blur、noise、temporal crop、frame dropping、frame duplication、frame-rate resampling、frame interpolation、video editing、VAE reconstruction、video-to-video regeneration、flow resampling 和 endpoint-preserving trajectory perturbation。

### 1.3 检测设定

SSTW 明确区分三种检测设定：

| 设定 | 可用信息 | 论文定位 | 是否作为主设定 |
|---|---|---|---|
| Owner-side trajectory audit | 生成端认证 trajectory sketch、密钥 \(K\)、模型 \(G_\theta\)、生成元数据 | 高置信服务端溯源与生成记录审计 | 是，但不得作为唯一主证据 |
| Model-side replay verification | 攻击后视频 \(\tilde{Y}\)、模型 \(G_\theta\)、密钥 \(K\)、可选 prompt \(c\)，通过 inversion / replay 估计轨迹 | 半白盒权属验证与鲁棒水印检测 | 是，主论文必须重点报告 |
| Video-only proxy detection | 仅攻击后视频 \(\tilde{Y}\) 和密钥 \(K\)，通过 video latent proxy 估计弱轨迹证据 | 黑盒退化检测 | 否，仅作为附录或扩展 |

主论文应以 **model-side replay verification** 作为核心鲁棒性设定，以 **owner-side trajectory audit** 作为高置信部署设定。除非后续实验能够证明 video-only proxy 在 fixed-FPR 下稳定成立，否则不得声明完全黑盒 Flow trajectory detection。

---

## 2. 方法总览

SSTW 的生成与检测流程如下：

```text
key K, prompt c, seed s
-> flow_tubelet_key_code
-> endpoint-aware minimum-energy flow control
-> velocity-field weak watermark constraint during Flow Matching sampling
-> watermarked Flow Matching trajectory {z_i, v'_i}
-> authenticated trajectory sketch and endpoint latent z_N
-> decoded video Y
-> attacked video observation Y_tilde
-> cached sketch retrieval or replay / inversion trajectory estimation
-> endpoint evidence extraction
-> velocity evidence extraction
-> time-reparameterization-invariant path evidence extraction
-> replay-uncertainty-aware flow-state inference
-> flow-state evidence admissibility
-> fixed low-FPR calibrated detection
```

SSTW 使用同一个密钥条件 tubelet code 在四个观测空间中形成同源证据：

| 证据层 | 观测对象 | 作用 | 是否可单独触发 positive |
|---|---|---|---|
| Endpoint evidence | 终点 latent 或重建 latent \(z_N\) | 提供 payload 基础证据 | 否 |
| Velocity evidence | 采样速度场或相邻采样状态位移 | 证明水印进入生成动力学 | 否 |
| Path evidence | 时间重参数化不变路径积分 | 证明跨采样时间的轨迹一致性 | 否 |
| Replay posterior evidence | replay / inversion 后验状态 | 证明攻击后视频仍保留轨迹后验结构 | 否 |

最终 positive 必须由四类证据通过 `flow_state_evidence_admissibility` 后进入 fixed-FPR calibrated detection。任何单一证据层均不得绕过校准阈值直接触发 positive。

---

## 3. 算法原语一：Flow Tubelet Key Code

### 3.1 Flow tubelet 划分

将视频 latent 轨迹划分为时空 tubelet：

\[
g=(\tau,u,v),
\]

其中，\(\tau\) 为时间 tubelet 索引，\((u,v)\) 为空间 tubelet 索引。对于采样步 \(i\)，tubelet 子块记为：

\[
z_{i,g}\in\mathbb{R}^{F_g\times C\times H_g\times W_g}.
\]

### 3.2 Flow-aware key direction

由密钥 \(K\)、tubelet 索引 \(g\)、采样时间 \(t_i\)、文本条件摘要 \(\bar{c}\) 和 sampler signature \(\sigma_{\mathrm{sampler}}\) 生成速度场投影方向：

\[
u_{i,g}(K)=\mathrm{Norm}\left(\mathrm{PRF}(K,g,t_i,\bar{c},\sigma_{\mathrm{sampler}})\right).
\]

其中，\(\sigma_{\mathrm{sampler}}\) 只用于生成端绑定采样协议；在 replay verification 中，应通过 sampler-mismatch control 检验该绑定是否过强。若 sampler 绑定导致泛化能力下降，应降级为：

\[
u_{i,g}(K)=\mathrm{Norm}\left(\mathrm{PRF}(K,g,\varphi(t_i),\bar{c})\right),
\]

其中，\(\varphi(t_i)\) 是归一化 flow phase，而非具体离散采样步编号。

### 3.3 Payload code

payload bit 定义为：

\[
b_g\in\{-1,+1\}.
\]

endpoint 约束目标为：

\[
\left\langle z_{N,g},u_{N,g}(K)\right\rangle\approx m_g b_g,
\]

其中，\(m_g\) 是 tubelet-level margin，可由内容复杂度、运动强度、质量门控和 endpoint controllability 决定。

### 3.4 Flow phase code

定义 flow phase code：

\[
\pi_{i,g}(K)=\phi_K(g,\varphi(t_i)),
\]

其中，\(\varphi(t_i)\in[0,1]\) 表示归一化 flow phase。该 code 的作用不是给帧编号，而是对不同采样区间的速度场扰动相位进行密钥条件调度。

### 3.5 Joint flow tubelet code

最终 tubelet code 为：

\[
c_{i,g}(K)=b_g\cdot\pi_{i,g}(K).
\]

该 code 同时服务于 velocity constraint、path-integral observation、endpoint evidence 和 state posterior inference。

---

## 4. 算法原语二：Velocity-Field Weak Watermark Constraint

### 4.1 主嵌入机制

SSTW 的主嵌入机制是 Flow Matching 采样时的速度场弱约束。对每个采样步 \(i\)，定义水印速度增量：

\[
\delta v_{i,g}
=
\lambda(t_i)\,a_{i,g}\,c_{i,g}(K)\,P_g u_{i,g}(K),
\]

其中，\(P_g\) 是 tubelet 局部投影算子，\(a_{i,g}\) 是内容与运动自适应强度，\(\lambda(t_i)\) 是采样时间调度函数。

水印后的速度场为：

\[
v'_\theta(z_i,t_i,c,K)
=
v_\theta(z_i,t_i,c)+\sum_g \delta v_{i,g}.
\]

对应采样更新为：

\[
z_{i+1}=z_i+\Delta t_i v'_\theta(z_i,t_i,c,K)+\mathcal{E}_i.
\]

### 4.2 时间调度

水印不应在全部采样区间均匀加入。采用中段主导的时间调度：

\[
\lambda(t)=
\lambda_0\cdot
\mathbb{I}[t\in(t_l,t_r)]\cdot
\sin^2\left(\pi\frac{t-t_l}{t_r-t_l}\right),
\]

其中，推荐 \(t_l\in[0.15,0.25]\)，\(t_r\in[0.75,0.85]\)。早期采样阶段主要决定全局语义，末期采样阶段主要决定视觉细节，二者均不宜承载过强水印。

### 4.3 内容、运动与可控性自适应强度

定义 tubelet 强度：

\[
a_{i,g}=\mathrm{clip}\left(
\alpha_0
\cdot M^{\mathrm{texture}}_{g}
\cdot M^{\mathrm{motion}}_{g}
\cdot M^{\mathrm{semantic-safe}}_{g}
\cdot M^{\mathrm{stability}}_{i,g}
\cdot M^{\mathrm{controllability}}_{i,g},
0,a_{\max}
\right).
\]

其中：

- \(M^{\mathrm{texture}}_{g}\) 表示纹理复杂度，纹理区域更适合承载弱扰动。
- \(M^{\mathrm{motion}}_{g}\) 表示局部运动强度，稳定运动区域更适合作为跨帧证据载体。
- \(M^{\mathrm{semantic-safe}}_{g}\) 表示语义安全区域，避免在人脸、文字、主体边界等敏感区域注入强扰动。
- \(M^{\mathrm{stability}}_{i,g}\) 表示采样过程中的局部稳定性。
- \(M^{\mathrm{controllability}}_{i,g}\) 表示当前速度场扰动对 endpoint response 的可控性。

### 4.4 质量保护约束

为了避免水印扰动破坏视频质量，需要满足全局能量预算：

\[
\sum_{i,g}\Delta t_i^2\left\|\delta v_{i,g}\right\|_2^2
\leq
B_{\mathrm{flow}}.
\]

同时要求局部速度扰动比例受限：

\[
\frac{\left\|\delta v_{i,g}\right\|_2}{\left\|v_\theta(z_i,t_i,c)_g\right\|_2+\epsilon}
\leq
r_{\max}.
\]

如果该约束被违反，当前 tubelet 不参与水印注入。

### 4.5 语义保持投影

为降低对 prompt semantics 的影响，将水印方向投影到近似语义低敏空间：

\[
\tilde{u}_{i,g}(K)=
\Pi_{\perp \mathcal{S}_{i,g}} u_{i,g}(K),
\]

其中，\(\mathcal{S}_{i,g}\) 表示由 cross-attention、text-video alignment gradient 或语义显著性 proxy 张成的局部敏感子空间。最终使用：

\[
u_{i,g}(K)\leftarrow\mathrm{Norm}(\tilde{u}_{i,g}(K)).
\]

该模块不要求首版必须依赖精确梯度实现，可先使用 attention map、saliency map、VAE latent energy 或 video-text saliency proxy 作为近似。

---

## 5. 算法原语三：Endpoint-Aware Minimum-Energy Flow Control

### 5.1 设计动机

仅在速度场中加入弱扰动不能保证 endpoint evidence 稳定保留。为避免水印变成“只在轨迹中可读、终点不可读”的审计信号，SSTW 引入 endpoint-aware minimum-energy flow control，将速度场约束、路径积分证据和终点 payload 统一到同一控制目标中。

### 5.2 受约束优化目标

对选定 tubelet，定义最小能量控制问题：

\[
\min_{\{\delta v_{i,g}\}}
\sum_{i,g}\Delta t_i^2\left\|\delta v_{i,g}\right\|_2^2,
\]

约束为：

\[
\left\langle z_{N,g},u_{N,g}(K)\right\rangle
\geq
m_g b_g,
\]

\[
\frac{\left\|\delta v_{i,g}\right\|_2}{\left\|v_\theta(z_i,t_i,c)_g\right\|_2+\epsilon}
\leq r_{\max},
\]

\[
\sum_{i,g}\Delta t_i^2\left\|\delta v_{i,g}\right\|_2^2
\leq B_{\mathrm{flow}}.
\]

实际实现中不要求求解完整最优控制问题，可使用一阶近似估计 endpoint controllability：

\[
M^{\mathrm{controllability}}_{i,g}
\propto
\left|
\left\langle
\frac{\partial \langle z_{N,g},u_{N,g}(K)\rangle}{\partial v_{i,g}},
P_g u_{i,g}(K)
\right\rangle
\right|.
\]

若梯度不可用，可使用 finite-difference probe 或历史 pilot 中的 endpoint response gain 近似。

### 5.3 论文主张边界

该模块的主张不是“严格最优控制”，而是：

```text
SSTW 使用 endpoint-aware minimum-energy approximation，使速度场扰动优先作用于对 endpoint payload 可控且对语义质量低敏的 tubelet。
```

必须通过以下消融证明其有效性：

```text
without_endpoint_aware_control
endpoint_agnostic_velocity_constraint
random_controllability_gate
finite_difference_controllability_gate
full_endpoint_aware_control
```

---

## 6. 算法原语四：Time-Reparameterization-Invariant Path Observation

### 6.1 轨迹响应

对于水印轨迹，定义速度场响应：

\[
r^{\mathrm{vel}}_{i,g}
=
\left\langle v'_\theta(z_i,t_i,c,K)_g,u_{i,g}(K)\right\rangle.
\]

定义路径增量响应：

\[
r^{\mathrm{path}}_{i,g}
=
\left\langle z_{i+1,g}-z_{i,g},u_{i,g}(K)\right\rangle.
\]

二者应满足：

\[
r^{\mathrm{path}}_{i,g}
\approx
\Delta t_i r^{\mathrm{vel}}_{i,g}.
\]

### 6.2 原始路径积分统计量

对每个 tubelet 计算原始路径积分证据：

\[
I^{\mathrm{raw}}_g(K)
=
\sum_{i=0}^{N-1}
\omega_i c_{i,g}(K) r^{\mathrm{path}}_{i,g}.
\]

该统计量适用于固定 sampler 与固定 time grid 的 cached trajectory audit，但对 wrong sampler replay 和 wrong time grid replay 较敏感。

### 6.3 时间重参数化不变路径积分

为增强跨 sampler、跨 step number 和跨 time grid 的稳定性，定义归一化路径积分：

\[
I^{\mathrm{inv}}_g(K)
=
\sum_{i=0}^{N-1}
\beta_i c_{i,g}(K)
\frac{
\left\langle z_{i+1,g}-z_{i,g},u_{i,g}(K)\right\rangle
}{
\left\|z_{i+1,g}-z_{i,g}\right\|_2+\epsilon
}.
\]

其中，\(\beta_i\) 是与归一化 flow phase 对齐的权重。该统计量近似衡量轨迹方向与密钥条件方向的一致性，而非具体步长大小，因此对时间重参数化更稳。

总体路径证据为：

\[
S_{\mathrm{path}}
=
\frac{1}{|\mathcal{G}'|}
\sum_{g\in\mathcal{G}'}
\mathrm{NormScore}(I^{\mathrm{inv}}_g(K)),
\]

其中，\(\mathcal{G}'\) 是通过质量门控、coverage gate 和 replay reliability gate 的有效 tubelet 集合。

### 6.4 终点一致性

endpoint payload response 定义为：

\[
r^{\mathrm{end}}_g
=
\left\langle \hat{z}_{N,g},u_{N,g}(K)\right\rangle.
\]

其中，\(\hat{z}_N\) 可来自生成缓存、VAE reconstruction 或 inversion。路径积分与终点响应应满足一致性：

\[
\mathrm{sign}(I^{\mathrm{inv}}_g(K))
\approx
\mathrm{sign}(r^{\mathrm{end}}_g b_g).
\]

定义 path-endpoint consistency：

\[
C_{\mathrm{path,end}}
=
\frac{1}{|\mathcal{G}'|}
\sum_{g\in\mathcal{G}'}
\mathbb{I}
\left[
\mathrm{sign}(I^{\mathrm{inv}}_g(K))=
\mathrm{sign}(r^{\mathrm{end}}_g b_g)
\right].
\]

### 6.5 路径证据的独立性要求

Path evidence 必须证明：

1. 在 \(H_0\) 与 \(H_1\) 之间存在统计分离。
2. 与 endpoint evidence 不高度冗余。
3. 在 fixed-FPR 下提供边际 TPR 增益。
4. trajectory controls 不能复现主增益。
5. wrong sampler、wrong prompt 和 wrong time grid 下仍有可解释退化曲线。

---

## 7. 轨迹观测来源与认证 sketch

### 7.1 Cached trajectory observation

在服务端生成场景中，系统保存压缩后的轨迹摘要，而不保存完整 \(z_i\)：

\[
\mathcal{T}_{\mathrm{sketch}}
=
\{\langle z_{i,g},u_{i,g}(K)\rangle,
\langle v_{i,g},u_{i,g}(K)\rangle,
I^{\mathrm{inv}}_g(K)\}_{i,g\in\Omega}.
\]

### 7.2 Authenticated trajectory sketch

为避免 owner-side trajectory audit 被质疑为普通日志记录或可伪造缓存，SSTW 使用服务端认证 sketch：

\[
\mathcal{C}_{\mathrm{traj}}
=
\mathrm{Sign}_{\mathrm{server}}
\left(
\mathrm{Hash}
(K_{\mathrm{id}},\mathcal{T}_{\mathrm{sketch}},c,s,t_{\mathrm{gen}},\sigma_{\mathrm{model}},\sigma_{\mathrm{sampler}})
\right).
\]

该模块不是论文的密码学主贡献，而是部署协议。论文中应将其定位为：

```text
authenticated trajectory sketch prevents owner-side audit from degenerating into unverifiable logging.
```

### 7.3 Replayed trajectory observation

对于攻击后视频，先通过 inversion 或 VAE reconstruction 得到 \(\hat{z}_N\)，再以 \(\hat{z}_N\)、prompt \(c\) 和模型 \(G_\theta\) 进行 reverse / replay 估计轨迹：

\[
\{\hat{z}_i,\hat{v}_i\}_{i=0}^{N}.
\]

该轨迹用于半白盒权属验证。论文主实验必须报告 replay error 对检测性能的影响。

### 7.4 Video-only proxy observation

当模型和 prompt 不可用时，只能从 \(\tilde{Y}\) 中提取 video latent proxy 和 temporal feature proxy。该设定只作为弱检测扩展，不作为主方法主张。

### 7.5 replay/sketch gate

`replay/sketch gate` 是连接 replay posterior evidence 与 authenticated trajectory sketch 的方法级门禁。它不是新的最终检测分数, 而是 Claim-3 的证据准入规则: 只有 authenticated sketch、replay uncertainty、wrong sampler replay、wrong prompt replay 和 replay negative FPR 同时闭合时, model-side replay verification 才能作为强 supported claim。

该 gate 的输入包括:

```text
authenticated trajectory sketch
replayed trajectory observation
replay uncertainty records
wrong sampler replay control
wrong prompt replay control
replay negative calibration records
```

该 gate 的输出包括:

```text
trajectory_sketch_verification_status
replay_or_sketch_status
replay_and_sketch_gate_decision
claim_support_status
```

如果 `replay/sketch gate` 未通过, 当前 `probe_paper`、`pilot_paper` 或 `full_paper` profile 必须失败。正式流程不提供 Claim-3 降级通道, 因为三个 profile 都要求同构地闭合三层主张。

---

## 8. 算法原语五：Replay-Uncertainty-Aware Flow-State Inference

### 8.1 局部观测向量

对每个时间位置或 tubelet group 构造观测：

\[
x_t=
[
 r^{\mathrm{end}}_t,
 r^{\mathrm{sync}}_t,
 r^{\mathrm{vel}}_t,
 r^{\mathrm{path}}_t,
 C^{\mathrm{path,end}}_t,
 q_t,
 \sigma^{\mathrm{replay}}_t,
 e_K(t)
].
\]

其中：

- \(r^{\mathrm{end}}_t\)：endpoint payload 响应。
- \(r^{\mathrm{sync}}_t\)：显式时间同步响应，仅作为观测项。
- \(r^{\mathrm{vel}}_t\)：速度场投影响应。
- \(r^{\mathrm{path}}_t\)：时间重参数化不变路径响应。
- \(C^{\mathrm{path,end}}_t\)：路径与终点一致性。
- \(q_t\)：质量、coverage 或 inversion reliability。
- \(\sigma^{\mathrm{replay}}_t\)：replay / inversion 不确定性。
- \(e_K(t)\)：密钥条件时间嵌入。

### 8.2 隐状态定义

定义 Flow watermark state：

\[
h_t=[\tau_t,e_t,\gamma_t,\epsilon_t,\rho_t,\kappa_t,\chi_t,\upsilon_t].
\]

| 状态变量 | 含义 | 主要失效模式 |
|---|---|---|
| \(\tau_t\) | 水印相位 | temporal crop、local clip |
| \(e_t\) | endpoint payload evidence | compression、VAE reconstruction |
| \(\gamma_t\) | posterior confidence | negative leakage、spurious match |
| \(\epsilon_t\) | 时间扰动状态 | dropping、duplication、FPS resampling |
| \(\rho_t\) | path consistency | trajectory mismatch、regeneration |
| \(\kappa_t\) | velocity consistency | velocity-field perturbation、sampler mismatch |
| \(\chi_t\) | inversion / replay reliability | inversion error、prompt mismatch |
| \(\upsilon_t\) | time-grid reliability | wrong time grid、sampler discretization mismatch |

### 8.3 Replay uncertainty weighting

定义 replay reliability weight：

\[
w_t^{\mathrm{replay}}
=
\exp\left(
-\frac{(\sigma^{\mathrm{replay}}_t)^2}{\tau_{\mathrm{rep}}}
\right).
\]

路径积分证据在 replay 设定中改写为：

\[
I^{\mathrm{rep}}_g(K)
=
\sum_i
w_i^{\mathrm{replay}}
\beta_i
c_{i,g}(K)
\frac{
\left\langle \hat{z}_{i+1,g}-\hat{z}_{i,g},u_{i,g}(K)\right\rangle
}{
\left\|\hat{z}_{i+1,g}-\hat{z}_{i,g}\right\|_2+\epsilon
}.
\]

状态后验为：

\[
p(h_{1:T}\mid x_{1:T},K,\sigma^{\mathrm{replay}}_{1:T}).
\]

这样，replay uncertainty 不只是报告指标，而是直接影响 trajectory evidence 和 posterior inference。

### 8.4 状态转移

状态转移为：

\[
h_t=f_\psi(h_{t-1},x_t,e_K(t),w_t^{\mathrm{replay}})+\xi_t.
\]

其中，\(f_\psi\) 可以使用轻量 SSM、Kalman-like recurrent model、gated recurrent model 或 selective scan 模型实现。论文主张不应写成使用某个网络结构，而应写成 replay-uncertainty-aware key-conditioned Flow watermark state posterior inference。

### 8.5 Filtering 与 smoothing

在线 filtering：

\[
\hat{h}^{\rightarrow}_t
=
\mathrm{Filter}_\psi(\hat{h}^{\rightarrow}_{t-1},x_t,e_K(t),w_t^{\mathrm{replay}}).
\]

双向 smoothing：

\[
\hat{h}_t
=
\mathrm{Smooth}_\psi(\hat{h}^{\rightarrow}_t,\hat{h}^{\leftarrow}_t,w_t^{\mathrm{replay}}).
\]

smoothing 只能使用当前样本观测序列，不能访问 test split 标签、阈值或攻击强度选择信息。

### 8.6 状态分数

定义状态一致性分数：

\[
S_{\mathrm{state}}
=
\Psi(\hat{h}_{1:T},x_{1:T},K,\sigma^{\mathrm{replay}}_{1:T}).
\]

状态分数不能单独触发 positive，必须进入 `flow_state_evidence_admissibility`。

---

## 9. 算法原语六：Flow-State Evidence Admissibility

### 9.1 设计原则

Flow trajectory inference 会扩大搜索空间。如果没有 admissibility 约束，状态模型可能在 negative sample 中搜索出伪轨迹，从而抬高 false positive tail。因此，SSTW 使用比普通 payload safety 更强的 `flow_state_evidence_admissibility`。

### 9.2 Admissibility 条件

定义：

\[
\mathrm{Adm}_{\mathrm{flow},K}
=
\mathbb{I}
[
A_1\land A_2\land A_3\land A_4\land A_5\land A_6\land A_7\land A_8\land A_9
].
\]

其中：

\[
A_1: S_{\mathrm{end,state}}\geq\eta_{\mathrm{end}},
\]

\[
A_2: S_{\mathrm{path}}\geq\eta_{\mathrm{path}},
\]

\[
A_3: C_{\mathrm{path,end}}\geq c_{\mathrm{pe}},
\]

\[
A_4: C_{\mathrm{posterior}}\geq c_{\mathrm{post}},
\]

\[
A_5: R_{\mathrm{coverage}}\geq r_0,
\]

\[
A_6: H_{\mathrm{state}}\leq h_0,
\]

\[
A_7: L_{\mathrm{neg-tail}}\leq \ell_0,
\]

\[
A_8: \bar{w}^{\mathrm{replay}}\geq w_0,
\]

\[
A_9: C_{\mathrm{time-grid}}\geq c_{\mathrm{grid}}.
\]

含义如下：

- \(S_{\mathrm{end,state}}\)：endpoint payload 与 state posterior 的联合证据。
- \(S_{\mathrm{path}}\)：时间重参数化不变路径证据。
- \(C_{\mathrm{path,end}}\)：路径证据与终点证据的一致性。
- \(C_{\mathrm{posterior}}\)：状态后验置信度。
- \(R_{\mathrm{coverage}}\)：有效 tubelet 覆盖率。
- \(H_{\mathrm{state}}\)：状态后验熵。
- \(L_{\mathrm{neg-tail}}\)：calibration negative 尾部风险。
- \(\bar{w}^{\mathrm{replay}}\)：平均 replay reliability。
- \(C_{\mathrm{time-grid}}\)：time-grid reliability。

### 9.3 禁止绕过规则

如果 \(\mathrm{Adm}_{\mathrm{flow},K}=0\)，则：

```text
velocity score cannot trigger positive
path score cannot trigger positive
state score cannot trigger positive
endpoint score alone cannot trigger final positive under trajectory claim
cached trajectory sketch alone cannot trigger final positive without evidence consistency
```

最终 decision 必须来自 fixed-FPR calibrated score。

---

## 10. 算法原语七：Fixed Low-FPR Calibrated Detection

### 10.1 Calibration negative

校准负样本必须包含 clean negative、attacked negative、replay negative 和 sampler-mismatch negative：

\[
\mathcal{D}_{\mathrm{calib-neg}}
=
\mathcal{D}_{\mathrm{clean-neg}}
\cup
\mathcal{A}(\mathcal{D}_{\mathrm{clean-neg}})
\cup
\mathcal{R}(\mathcal{D}_{\mathrm{clean-neg}})
\cup
\mathcal{M}_{\mathrm{sampler}}(\mathcal{D}_{\mathrm{clean-neg}}),
\]

其中，\(\mathcal{R}\) 表示 replay / inversion 过程，\(\mathcal{M}_{\mathrm{sampler}}\) 表示 wrong sampler、wrong time grid 或 wrong prompt replay。加入这些负样本是必要的，因为 trajectory replay 可能在 negative sample 中制造伪响应。

### 10.2 校准分数

对各证据分数进行 calibration negative 标准化：

\[
\tilde{S}_j
=
\frac{S_j-\mu_j^{\mathrm{calib-neg}}}{\sigma_j^{\mathrm{calib-neg}}+\epsilon}.
\]

主分数采用保守融合：

\[
S_{\mathrm{final}}
=
\mathrm{Adm}_{\mathrm{flow},K}
\cdot
\min
\left(
\tilde{S}_{\mathrm{end,state}},
\tilde{S}_{\mathrm{path}},
\tilde{S}_{\mathrm{path,end}},
\tilde{S}_{\mathrm{replay}}
\right).
\]

该保守融合优先保证 low-FPR safety。如果实验证明加权融合在 fixed-FPR 下稳定，也可报告补充分数：

\[
S_{\mathrm{final}}^{\mathrm{weighted}}
=
\mathrm{Adm}_{\mathrm{flow},K}
\cdot
\left(
\alpha_1\tilde{S}_{\mathrm{end,state}}
+
\alpha_2\tilde{S}_{\mathrm{path}}
+
\alpha_3\tilde{S}_{\mathrm{velocity}}
+
\alpha_4\tilde{S}_{\mathrm{path,end}}
+
\alpha_5\tilde{S}_{\mathrm{replay}}
\right).
\]

主表建议使用 conservative score，weighted score 作为补充实验。

### 10.3 阈值

给定 target FPR \(\alpha\)，阈值为：

\[
\eta_\alpha
=
\mathrm{Quantile}_{1-\alpha}
\left(
S_{\mathrm{final}}(\mathcal{D}_{\mathrm{calib-neg}})
\right).
\]

test split 中不得更新：

```text
threshold
fusion weights
state gate parameters
trajectory gate parameters
attack parameters
replay parameters
sampler mismatch parameters
```

### 10.4 决策

最终检测决策为：

\[
\mathrm{decision}
=
\mathbb{I}[S_{\mathrm{final}}\geq\eta_\alpha].
\]

所有 TPR、FPR、AUC、bit accuracy、attack breakdown、claim audit 和 failure reasons 必须从 event records 与 frozen thresholds 重建。

---

## 11. 主算法

### 11.1 Watermarked Flow Matching Sampling

```text
Algorithm 1: SSTW Watermarked Flow Matching Sampling
Input: model G_theta, prompt c, seed s, key K, sampler grid {t_i}, strength config Lambda
Output: watermarked video Y, authenticated trajectory sketch C_traj, endpoint latent z_N

1. Initialize z_0 from seed s.
2. Generate flow tubelet key code c_{i,g}(K) for selected tubelets and sampling steps.
3. Estimate semantic-safe, motion, texture, stability and endpoint controllability gates.
4. For i = 0 to N-1:
   4.1 Compute base velocity v_i = v_theta(z_i, t_i, c).
   4.2 Select admissible tubelets using content, motion, semantic-safe, stability and controllability gates.
   4.3 Compute watermark velocity increment delta v_{i,g}.
   4.4 Enforce local velocity ratio and global flow energy budget.
   4.5 Update v'_i = v_i + sum_g delta v_{i,g}.
   4.6 Update z_{i+1} = z_i + Delta t_i * v'_i.
   4.7 Store projection sketch instead of full trajectory when required.
5. Decode Y = D(z_N).
6. Compute authenticated trajectory sketch C_traj.
7. Return Y, C_traj, z_N.
```

### 11.2 SSTW Detection

```text
Algorithm 2: SSTW Detection
Input: attacked video Y_tilde, key K, optional prompt c, optional model G_theta, optional authenticated trajectory sketch C_traj, frozen thresholds
Output: decision, calibrated score, evidence report

1. Reconstruct or retrieve endpoint latent z_hat_N.
2. Obtain trajectory observation:
   2.1 verify and use authenticated trajectory sketch if available;
   2.2 otherwise run model-side replay / inversion;
   2.3 optionally use video-only proxy as fallback.
3. Estimate replay uncertainty and time-grid reliability.
4. Extract endpoint evidence, velocity evidence, time-reparameterization-invariant path evidence and path-endpoint consistency.
5. Build observation sequence x_{1:T}.
6. Run replay-uncertainty-aware key-conditioned flow-state inference.
7. Compute S_end,state, S_path, S_velocity, S_path,end, S_replay and posterior diagnostics.
8. Evaluate Adm_{flow,K}.
9. Compute S_final using frozen calibration statistics.
10. Compare S_final with frozen threshold eta_alpha.
11. Return decision and full evidence report.
```

---

## 12. 统计分离命题

### 12.1 Path evidence 的期望分离

在 \(H_0\) 下，由于 \(u_{i,g}(K)\) 为密钥条件伪随机方向，并且 negative video 的 trajectory increment 与正确密钥方向无系统相关性，有：

\[
\mathbb{E}_{H_0}[I^{\mathrm{inv}}_g(K)]\approx 0.
\]

在 \(H_1\) 下，由于速度场注入项与 \(c_{i,g}(K)u_{i,g}(K)\) 对齐，有：

\[
\mathbb{E}_{H_1}[I^{\mathrm{inv}}_g(K)]
\approx
\sum_i
\beta_i
\frac{
\Delta t_i\lambda(t_i)a_{i,g}
}{
\|z_{i+1,g}-z_{i,g}\|_2+\epsilon
}
>0.
\]

因此，路径证据的期望分离为：

\[
\Delta_{\mathrm{path}}
=
\mathbb{E}_{H_1}[S_{\mathrm{path}}]
-
\mathbb{E}_{H_0}[S_{\mathrm{path}}].
\]

SSTW 的核心实验必须证明：

\[
\Delta_{\mathrm{path}}>0
\quad\text{and}\quad
\mathrm{TPR}_{\mathrm{with\ path}}@\mathrm{FPR}=\alpha
>
\mathrm{TPR}_{\mathrm{endpoint\ only}}@\mathrm{FPR}=\alpha.
\]

### 12.2 False positive tail 控制

由于状态搜索和 replay 会扩大隐式假设空间，SSTW 不直接使用 raw path score 判定，而是通过 calibration negative、admissibility 和 conservative fusion 控制 tail risk：

\[
P_{H_0}(S_{\mathrm{final}}\geq\eta_\alpha)\leq\alpha.
\]

该不等式必须通过 held-out test split 的 clean negative、attacked negative、replay negative 和 sampler-mismatch negative 实证验证。

---

## 13. 必须对比的 baseline

### 13.1 普通视频水印 baseline

```text
framewise_payload_watermark
segment_payload_watermark
posthoc_video_watermark_decoder
video_vae_latent_watermark
```

### 13.2 显式时间对齐 baseline

```text
explicit_tubelet_sync
sliding_window_temporal_matching
DTW_temporal_alignment
edit_distance_temporal_matching
segment_group_ordering
```

### 13.3 轨迹水印 baseline

```text
endpoint_latent_only
initial_noise_only
velocity_projection_without_path_integral
path_integral_without_velocity_constraint
trajectory_score_late_fusion
```

### 13.4 状态模型 baseline

```text
no_state_inference
temporal_mean_pooling
conv1d_temporal_aggregator
gru_temporal_aggregator
transformer_temporal_aggregator
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_without_flow_trajectory
key_conditioned_flow_state_inference
replay_uncertainty_aware_flow_state_inference
```

### 13.5 Flow-specific controls

```text
flow_time_shuffle
flow_key_shuffle
flow_random_direction
flow_wrong_prompt_replay
flow_wrong_sampler_replay
flow_wrong_time_grid_replay
flow_without_velocity_constraint
flow_without_path_integral
flow_without_path_endpoint_consistency
flow_without_replay_uncertainty
```

---

## 14. 必须消融

### 14.1 Watermark injection 消融

```text
no_watermark
endpoint_only_watermark
velocity_only_watermark
path_integral_only_observation
velocity_constraint_without_content_adaptation
velocity_constraint_without_semantic_safe_projection
velocity_constraint_without_endpoint_control
full_fm_sstw
```

### 14.2 时间调度消融

```text
uniform_lambda
early_only_lambda
middle_only_lambda
late_only_lambda
sinusoidal_middle_lambda
learned_lambda_if_available
```

### 14.3 最小能量控制消融

```text
without_endpoint_aware_control
endpoint_agnostic_velocity_constraint
random_controllability_gate
finite_difference_controllability_gate
gradient_controllability_gate
full_endpoint_aware_minimum_energy_control
```

### 14.4 路径证据消融

```text
path_integral_raw
path_integral_time_normalized
path_integral_reparameterization_invariant
path_integral_without_endpoint_consistency
path_integral_without_replay_weight
```

### 14.5 密钥条件消融

```text
correct_key
shuffled_key
random_key_embedding
key_agnostic_flow_state
key_conditioned_endpoint_only
key_conditioned_velocity_and_path
```

### 14.6 轨迹观测消融

```text
cached_trajectory
replayed_trajectory
video_only_proxy
trajectory_time_shuffle
trajectory_key_shuffle
trajectory_random_direction
trajectory_wrong_sampler
trajectory_wrong_prompt
trajectory_wrong_time_grid
```

### 14.7 Admissibility 消融

```text
with_flow_state_admissibility
without_admissibility
endpoint_only_admissibility
path_only_admissibility
state_only_admissibility
without_negative_tail_gate
without_path_endpoint_consistency_gate
without_replay_reliability_gate
without_time_grid_reliability_gate
```

---

## 15. 攻击协议

### 15.1 空间攻击

```text
JPEG / video codec compression
resize
center crop
random crop
Gaussian noise
Gaussian blur
color jitter
VAE reconstruction
```

### 15.2 时间攻击

```text
temporal crop
local clip
frame dropping
frame duplication
frame insertion
frame reordering
frame-rate resampling
frame interpolation
temporal smoothing
```

### 15.3 生成式攻击

```text
video-to-video regeneration
diffusion / flow resampling
latent inversion and reconstruction
prompt-preserving video editing
motion editing
style transfer
VAE encode-decode chains
```

### 15.4 Flow-specific 自适应攻击

```text
velocity_projection_removal
path_integral_cancellation
endpoint_preserving_flow_resampling
velocity_randomization_with_endpoint_reconstruction
flow_nullspace_trajectory_perturbation
multi_query_watermark_direction_estimation
sampler_mismatch_attack
time_grid_mismatch_attack
trajectory_replay_disruption
trajectory_sketch_forgery_without_key
detector_probing_with_public_negatives
```

所有攻击均需在 clean negative、attacked negative、replay negative 和 watermarked attacked positive 上同时报告，以避免只展示 positive robustness 而忽略 FPR tail。

---

## 16. 评价指标

### 16.1 水印检测指标

```text
TPR at FPR = 0.01
TPR at FPR = 0.001
TPR at FPR = 0.0001 if sample size allows
clean negative FPR
attacked negative FPR
replay negative FPR
sampler-mismatch negative FPR
AUC
bit accuracy
message recovery accuracy
log p-value if applicable
```

### 16.2 轨迹证据指标

```text
velocity response separation
raw path-integral response separation
time-reparameterization-invariant path response separation
path-endpoint consistency
trajectory marginal gain under fixed-FPR
trajectory-payload redundancy
replay error sensitivity
state posterior entropy
negative tail inflation
time-grid reliability curve
sampler mismatch degradation curve
```

### 16.3 视频质量指标

```text
FVD
LPIPS
SSIM
PSNR
CLIP / video-text alignment
motion consistency
warping error
temporal flicker
human perceptual study if available
```

### 16.4 效率指标

```text
generation overhead
detection overhead
trajectory sketch storage
memory overhead
runtime under different video lengths
runtime under different sampling steps
runtime under different replay settings
```

---

## 17. 论文主张与 claim audit

| 论文主张 | 必须证据 | 必须 control |
|---|---|---|
| Flow Matching 速度场弱约束能形成可检测水印 | endpoint 与 path evidence 均分离 | endpoint_only、velocity_only、no_watermark |
| 最小能量 Flow 控制降低质量损失 | 同等 TPR 下 FVD、motion、semantic 更稳 | without_endpoint_aware_control、random_controllability_gate |
| 时间重参数化不变 path evidence 更稳 | wrong sampler / wrong grid 下优于 raw path | path_integral_raw、wrong_time_grid_replay |
| Path evidence 提供独立增益 | fixed-FPR 下加入 path evidence 后 TPR 提升 | trajectory_time_shuffle、trajectory_key_shuffle、random_direction |
| 状态空间后验推断优于显式对齐 | temporal attacks 下优于 DTW、edit-distance、segment ordering | generic SSM、GRU、Transformer |
| Replay uncertainty 是必要因素 | 去掉后 replay negative FPR 或 tail risk 上升 | without_replay_uncertainty、wrong_prompt、wrong_sampler |
| Key condition 是必要因素 | 正确 key 显著优于 shuffled / random key | key_agnostic_state_space |
| Admissibility 抑制 false positive tail | 去掉后 attacked / replay negative FPR 上升 | without_admissibility、without_negative_tail_gate |
| Owner-side sketch 可认证 | sketch 不可被无 key 伪造为有效证据 | trajectory_sketch_forgery_without_key |
| 质量保持成立 | FVD、LPIPS、motion consistency 不明显下降 | unwatermarked generation、endpoint_only |

supported claim 必须映射到 governed records、tables、figures、reports 或 manifests。任何没有 records 支撑的结论只能进入 limitation 或 future work。

---

## 18. 三层论文主张策略

为降低审稿风险，SSTW 的论文主张分为三层：

| 层级 | 主张 | 必须成立条件 | 投稿写法 |
|---|---|---|---|
| Claim-1 | Flow velocity constraint 能形成低扰动 endpoint watermark | endpoint evidence 分离，质量不明显下降 | 必须作为主贡献 |
| Claim-2 | Path-integral evidence 在 fixed-FPR 下提供独立增益 | path evidence 非冗余，TPR@FPR 提升 | 必须作为主贡献 |
| Claim-3 | Replay verification 可从攻击后视频恢复轨迹后验 | replay/sketch gate 通过, replay error 可控，model-side verification 成立 | 必须作为三层闭合主贡献; gate 未通过则 profile 失败 |

Claim-1、Claim-2 与 Claim-3 必须同时成立, 否则当前正式 paper profile 必须失败。owner-side audit 可以作为补充解释, 但不能替代攻击后视频 replay posterior、认证 sketch 和固定 FPR 控制共同形成的 Claim-3 正式证据。

---

## 19. 失败边界与降级策略

### 19.1 Velocity constraint 失败

如果速度场弱约束导致明显视频质量下降、motion artifact 或 semantic inconsistency，则不能将其作为主方法贡献。此时应降低 \(\lambda(t)\)、缩小 tubelet coverage、增强 semantic-safe projection，或降级为 endpoint latent watermark + trajectory observation。

### 19.2 Endpoint-aware control 失败

如果 endpoint-aware minimum-energy control 不能改善 endpoint evidence 或质量指标，则不得作为主贡献。此时保留 velocity-field weak constraint，并将该模块作为附录探索。

### 19.3 Path evidence 失败

如果 \(S_{\mathrm{path}}\) 与 endpoint evidence 高度冗余，或在 fixed-FPR 下不能提供边际增益，则 SSTW 不应作为主论文核心贡献。此时可降级为状态空间同步视频水印或 endpoint-aware Flow latent watermark。

### 19.4 State inference 失败

如果 replay-uncertainty-aware flow-state inference 不优于 explicit temporal alignment、generic SSM、GRU 或 Transformer aggregator，则状态空间主张不成立。此时需要重新设计状态变量、观测模型和 key embedding，而不是单纯增加模型容量。

### 19.5 Replay verification 失败

如果 replay / inversion 误差过大导致轨迹证据不可用，则主设定应限制为 owner-side trajectory audit，并将 model-side replay verification 降级为附录探索。

### 19.6 Admissibility 失败

如果 admissibility 显著降低 TPR 且不能降低 attacked negative FPR、replay negative FPR 或 tail risk，则说明 gate 设计不合理，需要重新校准 coverage、posterior confidence、replay reliability 和 negative-tail 约束。

---

## 20. 最终方法边界

SSTW 的最终方法边界是：

\[
\boxed{
\text{Flow Matching velocity-field watermarking and replay-aware path-posterior inference are the method.}
}
\]

中文表述为：

\[
\boxed{
\text{Flow Matching 速度场水印约束与 replay 感知路径后验推断是方法主体。}
}
\]

显式时间对齐、普通 tubelet sync、frame-wise sequence matching、DTW、edit-distance matching、segment ordering、generic SSM、Mamba-style temporal fusion、通用时序聚合器和未认证 trajectory logging 均只能作为 baseline、control 或部署组件，不能作为 SSTW 的方法主体。

---

## 21. 最终贡献写法

如果所有核心机制成立，SSTW 的贡献应写为：

1. 本文将 Flow Matching 视频生成水印从终点 latent 检测问题重新定义为生成速度场、路径积分、终点证据与 replay 后验之间的密钥条件轨迹一致性推断问题。
2. 本文提出 Flow Tubelet Key Code，使 payload、flow phase、velocity constraint、path-integral observation、endpoint evidence 和 replay posterior inference 由同一个密钥条件 tubelet 结构统一生成。
3. 本文提出 Velocity-Field Weak Watermark Constraint，在 Flow Matching 采样过程中以能量受限、语义保护和中段调度的方式注入低扰动水印速度增量。
4. 本文提出 Endpoint-Aware Minimum-Energy Flow Control，使速度场扰动优先作用于对 endpoint payload 可控且对视频质量低敏的 tubelet。
5. 本文提出 Time-Reparameterization-Invariant Path Observation，将水印检测从最终视频帧扩展到生成路径，并增强不同 sampler、不同 step number 与不同 time grid 下的轨迹证据稳定性。
6. 本文提出 Replay-Uncertainty-Aware Flow-State Inference，递推估计水印相位、payload evidence、时间扰动、velocity consistency、path consistency、time-grid reliability 和 replay reliability。
7. 本文提出 Flow-State Evidence Admissibility，在 fixed low-FPR protocol 下约束状态搜索和轨迹观测，防止 trajectory score、cached sketch 或 state score 绕过 payload evidence 造成 false positive tail。
8. 本文在显式时间对齐、普通 temporal aggregator、generic SSM、key-agnostic SSM、endpoint-only watermark、trajectory-only score、wrong sampler replay 和 Flow-specific adaptive attacks 等 baseline 与 control 下完成系统消融，证明 SSTW 的核心增益来自 Flow Matching 轨迹水印机制，而不是模型复杂度、分数堆叠或服务端日志。
