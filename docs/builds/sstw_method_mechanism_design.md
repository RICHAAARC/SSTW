# SSTW 方法机制设计：面向生成式视频轨迹的密钥条件状态空间水印推断

## 0. 文档定位

本文档定义 SSTW 的方法机制。本文方法不是显式同步机制的升级版, 也不是在已有 tubelet 同步方法上添加 SSM 模块。本文的独立研究对象是:

\[
\boxed{
\text{Key-conditioned state-space inference for generative video watermark trajectories}
}
\]

方法主体由五个算法原语组成:

```text
key_conditioned_tubelet_code
key_conditioned_state_space_inference
generative_trajectory_observation
key_state_evidence_admissibility
fixed_low_fpr_calibrated_detection
```

其中 `generative_trajectory_observation` 是区分 SSTW 与普通真实视频同步水印工作的关键机制。若该机制不能成立, SSTW 不应按顶会主论文投稿。

---

## 1. 问题定义

### 1.1 生成式视频水印检测任务

给定视频生成模型 \(G_\theta\)、文本条件 \(c\)、随机种子 \(s\) 和密钥 \(K\), 生成过程产生视频 latent 轨迹:

\[
\{z_t\}_{t=0}^{T},\quad z_t\in\mathbb{R}^{F\times C\times H\times W}。
\]

最终视频为:

\[
Y = D(z_T),
\]

其中 \(D\) 是视频解码器或 VAE decoder。

攻击后的观测视频记为:

\[
\tilde{Y}=\mathcal{A}(Y),
\]

其中 \(\mathcal{A}\) 可以包含 compression、resize、temporal crop、frame dropping、frame duplication、frame-rate resampling 和 VAE reconstruction。

检测目标是二元假设检验:

\[
H_0:\tilde{Y}\sim P_{\mathrm{clean}},\quad
H_1:\tilde{Y}\sim P_{\mathrm{wm}}。
\]

### 1.2 核心困难

生成式视频水印检测面临三个非显然困难:

1. 时间攻击会破坏水印相位和 tubelet 对应关系。

2. 生成轨迹中的水印证据可能不完全体现在最终视频帧中。

3. 状态搜索会扩大负样本 tail, 若没有 fixed-FPR 校准与 admissibility 约束, 容易产生 false positive。

因此, 本文不把时间失同步建模为简单 offset / scale 对齐, 而建模为密钥条件隐状态估计问题。

---

## 2. 方法总览

SSTW 的方法从同一个 key-conditioned tubelet code 出发, 通过状态空间模型递推估计水印相位、局部证据、同步置信度、时间扰动和轨迹一致性。

整体流程为:

```text
key K, prompt c, seed s
-> key-conditioned tubelet code
-> watermark embedding or trajectory weak constraint
-> generated latent trajectory
-> attacked video observation
-> local tubelet evidence extraction
-> trajectory observation extraction
-> key-conditioned state-space inference
-> key-state evidence admissibility
-> fixed-FPR calibrated detection
```

核心要求是: payload evidence、temporal evidence 和 trajectory evidence 必须是同一个 key-conditioned tubelet code 在不同观测空间中的投影, 不能是互不相关的分数堆叠。

---

## 3. 算法原语一：Key-Conditioned Tubelet Code

### 3.1 Tubelet 划分

将视频 latent 划分为时空 tubelet:

\[
g=(\tau,u,v),
\]

其中 \(\tau\) 是时间块索引, \((u,v)\) 是空间块索引。每个 tubelet 对应 latent 子块:

\[
z_g\in\mathbb{R}^{F_g\times C\times H_g\times W_g}。
\]

### 3.2 密钥方向

由密钥 \(K\) 生成 tubelet 方向:

\[
u_g(K)=\mathrm{Norm}(\mathrm{PRF}(K,g))。
\]

该方向同时服务于 payload evidence、state observation 和 trajectory observation。

### 3.3 Payload code

payload bit \(b_g\in\{-1,+1\}\) 的投影约束为:

\[
\langle z_g,u_g(K)\rangle \approx m\cdot b_g。
\]

其中 \(m\) 是 margin 强度。

### 3.4 Synchronization code

同步相位 code 由密钥和 tubelet 时间索引共同决定:

\[
c^{\mathrm{sync}}_g = \phi_K(\tau)。
\]

### 3.5 Joint tubelet code

本文使用统一编码:

\[
c_g(K)=c^{\mathrm{payload}}_g(K)\cdot c^{\mathrm{sync}}_g(K)。
\]

这保证 payload、state 和 trajectory 不是独立设计的三组证据, 而是同一水印结构的不同观测。

---

## 4. 算法原语二：Key-Conditioned State-Space Inference

### 4.1 局部观测向量

对攻击后视频或重新编码 latent 提取局部观测:

\[
x_t=[r^{\mathrm{pay}}_t,r^{\mathrm{sync}}_t,q_t,e_K(t)]。
\]

其中:

- \(r^{\mathrm{pay}}_t\): payload 局部响应。

- \(r^{\mathrm{sync}}_t\): 同步局部响应。

- \(q_t\): 质量或可置信观测。

- \(e_K(t)\): key-conditioned temporal embedding。

若启用生成轨迹观测, 则扩展为:

\[
x_t=[r^{\mathrm{pay}}_t,r^{\mathrm{sync}}_t,r^{\mathrm{traj}}_t,q_t,e_K(t)]。
\]

### 4.2 隐状态定义

状态变量定义为:

\[
h_t=[\tau_t,e_t,\gamma_t,\epsilon_t,\rho_t]。
\]

| 状态变量 | 含义 | 对应失效模式 |

|---|---|---|

| \(\tau_t\) | 水印相位 | temporal crop / local clip |

| \(e_t\) | 局部 payload evidence | compression / VAE reconstruction |

| \(\gamma_t\) | 同步置信度 | negative leakage / spurious correlation |

| \(\epsilon_t\) | 时间扰动 | dropping / duplication / resampling |

| \(\rho_t\) | 轨迹一致性 | generation trajectory mismatch |

### 4.3 状态转移模型

状态转移写为:

\[
h_t = f_\theta(h_{t-1},x_t,e_K(t)) + \xi_t。
\]

其中 \(e_K(t)\) 是密钥条件项。它可以进入 transition、observation 或 smoothing 权重, 但必须在 ablation 中证明必要性。

### 4.4 观测模型

观测似然为:

\[
p(x_t\mid h_t,K)=\mathcal{O}_\theta(x_t,h_t,e_K(t))。
\]

状态估计目标为:

\[
p(h_{1:T}\mid x_{1:T},K)。
\]

### 4.5 Filtering 与 smoothing

在线 filtering:

\[
\hat{h}_t^{\rightarrow}=\mathrm{Filter}_\theta(\hat{h}_{t-1}^{\rightarrow},x_t,e_K(t))。
\]

双向 smoothing:

\[
\hat{h}_t=\mathrm{Smooth}_\theta(\hat{h}_t^{\rightarrow},\hat{h}_t^{\leftarrow})。
\]

smoothing 只能使用当前样本的观测序列, 不能访问 test split 标签或阈值选择信息。

### 4.6 状态空间分数

状态 posterior 产生状态一致性分数:

\[
S_{\mathrm{state}}=\Psi(\hat{h}_{1:T},x_{1:T},K)。
\]

该分数不能单独触发 positive, 必须通过 key-state evidence admissibility。

---

## 5. 算法原语三：Generative Trajectory Observation

### 5.1 轨迹来源

对于 DiT / Flow Matching 视频生成模型, 生成过程提供中间轨迹:

\[
\{z_t,v_\theta(z_t,t,c)\}_{t=0}^{T}。
\]

其中 \(v_\theta\) 是速度场或 denoising / flow 更新方向。

### 5.2 轨迹响应

对每个 tubelet 计算轨迹响应:

\[
r^{\mathrm{traj}}_t = \langle \Delta z_t, u_g(K)\rangle,
\]

或对于 Flow Matching 速度场:

\[
r^{\mathrm{traj}}_t = \langle v_\theta(z_t,t,c), u_g(K)\rangle。
\]

该响应不直接作为后验分数相加, 而作为状态观测项进入 \(x_t\)。

### 5.3 独立性要求

Trajectory observation 必须证明:

1. 在 \(H_0\) 与 \(H_1\) 之间存在统计分离。

2. 与 static payload evidence 不高度冗余。

3. 在 fixed-FPR 下提供边际增益。

4. trajectory controls 不能复现主增益。

5. runtime overhead 可接受。

### 5.4 Controls

必须包含以下 controls:

```text
trajectory_time_shuffle
trajectory_key_shuffle
trajectory_random_direction
trajectory_without_key_condition
static_tubelet_only_control
generic_state_space_with_trajectory_control
```

这些 controls 用于证明增益来自密钥条件生成轨迹观测, 而不是简单增加一个分数分支。

---

## 6. 算法原语四：Key-State Evidence Admissibility

### 6.1 基本原则

状态推断只能说明检测器找到了一条可能合理的水印状态轨迹, 不能单独证明水印存在。最终 positive 必须由 payload evidence、state posterior 和 fixed-FPR negative safety 共同支撑。

因此本文使用 `key_state_evidence_admissibility`, 而不使用容易被理解为普通 payload safety 的命名。

### 6.2 Admissibility 条件

\[
\mathrm{Adm}_{K}
=
\mathbb{I}[
S_{\mathrm{payload,state}}\ge\eta_{\mathrm{payload}}
\land
C_{\mathrm{posterior}}\ge c_0
\land
R_{\mathrm{coverage}}\ge r_0
\land
N_{\mathrm{matched}}\ge n_0
\land
H_{\mathrm{state}}\le h_0
\land
L_{\mathrm{neg-tail}}\le \ell_0
]。
\]

其中:

- \(C_{\mathrm{posterior}}\): posterior state confidence。

- \(R_{\mathrm{coverage}}\): 有效 tubelet 覆盖率。

- \(N_{\mathrm{matched}}\): 匹配状态数量。

- \(H_{\mathrm{state}}\): 状态熵。

- \(L_{\mathrm{neg-tail}}\): calibration negative tail 风险。

### 6.3 最终证据

最终分数为:

\[
S_{\mathrm{final}}
=
\mathcal{F}_{\mathrm{calib}}
(
S_{\mathrm{payload}},
S_{\mathrm{state}},
S_{\mathrm{trajectory}},
\mathrm{Adm}_{K}
)。
\]

若 \(\mathrm{Adm}_{K}=0\), 状态分数与轨迹分数不得绕过 payload evidence 直接触发 positive。

---

## 7. 算法原语五：Sampling-Time Weak Constraint

### 7.1 速度场弱约束

当 detector-side trajectory observation 已经成立后, 可以进一步探索 sampling-time weak constraint:

\[
v'_\theta(z_t,t,c)=v_\theta(z_t,t,c)+\lambda(t)\sum_g c_g(K)P_gu_g(K)。
\]

其中 \(P_g\) 是 tubelet 局部投影算子, \(\lambda(t)\) 是采样时间调度函数。

建议只在中间采样区间启用:

\[
\lambda(t)=0,\quad t\in[0,0.2]\cup[0.8,1]。
\]

### 7.2 成立条件

该模块只有同时满足以下条件才能作为强贡献:

1. 提升 \(S_{\mathrm{trajectory}}\)。

2. 提升 attacked positive TPR。

3. 不破坏 target FPR。

4. 视频质量下降可控。

5. motion consistency 不明显下降。

6. prompt / semantic consistency 不明显下降。

7. 不引入明显 motion artifact。

若任一条件不满足, 该模块应降级为附录探索。

---

## 8. Fixed Low-FPR Calibrated Detection

### 8.1 Calibration negative

校准负样本必须包含 clean negative 和 attacked negative:

\[
\mathcal{D}_{\mathrm{calib-neg}}
=
\mathcal{D}_{\mathrm{clean-neg}}
\cup
\mathcal{A}(\mathcal{D}_{\mathrm{clean-neg}})。
\]

这是因为状态搜索、双向 smoothing 和 trajectory observation 都会扩大隐式搜索空间。若不使用 attacked negative 校准, FPR 会被低估。

### 8.2 阈值

给定 target FPR \(\alpha\):

\[
\eta_\alpha=
\mathrm{Quantile}_{1-\alpha}(S_{\mathrm{final}}(\mathcal{D}_{\mathrm{calib-neg}}))。
\]

在 test split 中, 阈值、融合权重、状态门控参数和攻击参数都不得更新。

### 8.3 输出决策

\[
\mathrm{decision}=\mathbb{I}[S_{\mathrm{final}}\ge\eta_\alpha]。
\]

所有 TPR、FPR、AUC、attack breakdown 和 claim audit 都必须从 event records 与 thresholds 重建。

---

## 9. 方法主张

若所有核心机制成立, SSTW 的贡献应写为:

1. 将生成式视频水印中的 temporal desynchronization 形式化为密钥条件隐状态估计问题。

2. 提出 key-conditioned tubelet code, 使 payload、synchronization 与 trajectory observation 统一到同一 latent tubelet 结构。

3. 提出 key-conditioned state-space inference, 递推估计水印相位、局部证据、同步置信度、时间扰动和轨迹一致性。

4. 提出 key-state evidence admissibility, 防止状态搜索和轨迹观测绕过 payload evidence 造成 false positive tail。

5. 将 generative trajectory observation 作为状态观测项, 证明生成过程证据能提供最终视频帧之外的独立水印信息。

6. 在 fixed low-FPR protocol 下完成系统消融, 证明 key condition、state variables、trajectory observation 和 admissibility 的必要性。

7. 可选地探索 sampling-time weak constraint, 将检测端轨迹证据推进到生成过程弱约束。

---

## 10. 必须消融

### 10.1 状态模型消融

```text
no_state_inference
generic_temporal_mean_pooling
conv1d_temporal_aggregator
gru_temporal_aggregator
transformer_temporal_aggregator
generic_state_space_model
key_agnostic_state_space_model
key_conditioned_state_space_inference
```

### 10.2 Key condition 消融

```text
key_conditioned_state_space_inference
state_space_without_key_condition
state_space_with_shuffled_key
state_space_with_random_key_embedding
```

### 10.3 Trajectory 消融

```text
key_conditioned_state_space_inference
key_conditioned_state_space_with_trajectory
trajectory_only
trajectory_time_shuffle
trajectory_key_shuffle
trajectory_random_direction
```

### 10.4 Admissibility 消融

```text
with_key_state_admissibility
without_key_state_admissibility
payload_only_admissibility
state_only_admissibility
trajectory_only_admissibility
```

---

## 11. 方法失败与降级策略

### 11.1 状态空间推断失败

若 `key_conditioned_state_space_inference` 不能优于 explicit temporal alignment、generic SSM 或普通 temporal aggregator, 则不能将状态空间推断作为核心贡献。需要回到状态变量、key embedding 和 observation model 重新设计。

### 11.2 Trajectory observation 失败

若 trajectory response 与 payload evidence 高度冗余, 或 fixed-FPR 下没有边际增益, 则 SSTW 不应按顶会主论文投稿。此时最多形成技术报告或附录探索。

### 11.3 Sampling-time weak constraint 失败

若该模块影响视频质量、motion consistency 或 semantic consistency, 则不得写成主方法。主线应回到 state-space inference + trajectory observation。

---

## 12. 最终边界

SSTW 的最终边界是:

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
