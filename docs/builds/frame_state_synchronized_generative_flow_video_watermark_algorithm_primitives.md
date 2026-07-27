# 面向 Flow-Matching 视频 Transformer 的状态空间同步轨迹水印算法原语

## 0. 适用范围

本文只冻结方法语义和可证伪原语，不冻结尚未通过 construction 的具体数值、
threshold、状态维度或攻击网格。当前：

```text
formal_result=false
stage_progression_allowed=false
runtime_implementation_authorized=false
```

历史 Flow-stage carrier、prompt-conditioned replay、全局 VAE feature，以及
最新失败的 decoder-Jacobian additive atom/local-mean feature 均不得作为本算法
原语已成立的证据。后者否定的是替代 carrier/feature 组合，不是本文件定义的
Patch 关系动力学水印。

更精确地说，最新真实 Gate 0 FAIL 只否定
`decoder-Jacobian additive atom + local RGB mean feature + held-out transfer`
组合，不否定 Patch-relation embedding、observer 或状态空间同步。generic public
low-frequency carrier bank 只能作为待审 baseline/fallback。

本文与配套 method design 共同冻结的唯一正式机制链为：

```text
payload M
→ PRC drive u_n
→ low-dimensional state dynamics s_n
→ DiT Patch/3D-RoPE or relative-attention carrier
→ inference-time Flow velocity deflection
→ output-side Patch-relation observation
→ clock path + state observer
→ key-conditioned trajectory evidence
```

本合同不授权 GPU/Colab、runner、Notebook、Drive 或方法 runtime 实施，也不
授权攻击、fixed-FPR、baseline 执行、paper claim 或阶段推进。

## 1. 符号与双时间坐标

| 符号 | 含义 |
|---|---|
| \(\tau\) | 视频生成模型的 Flow/sampler 时间 |
| \(n\) | 原始水印视频帧或固定时间窗口索引 |
| \(i\) | 受攻击视频中的观测帧/窗口索引 |
| \(z_\tau\) | 视频模型高维生成 latent |
| \(s_n\) | 低维帧间水印状态 |
| \(M\) | 用户/归属 payload |
| \(u_n\) | PRC 编码后的密钥条件时间驱动 |
| \(p,q\) | DiT 视频 token 的空间/时间 Patch 索引 |
| \(\mathcal R_j\) | 第 \(j\) 个公开 Patch-pair/relative-relation atom |
| \(c_K(s_n)\) | 由密钥和状态唯一派生的公开 relation atom 系数 |
| \(\rho(n)\) | 视频状态窗口到冻结 relation block/token support 的公开映射 |
| \(\pi(i)\) | 观测索引到水印时钟的对齐路径 |
| \(q_i\) | 从最终视频提取的 Patch 关系局部观测 |

强制不变量：

```text
watermark_state_clock = video_frame_time
embedding_schedule_clock = generative_flow_time
carrier_coordinate = public_patch_relative_relation
```

任何代码不得使用同一个未区分的 `t` 同时表示两条时间轴。
“轨迹”不得含混：\(z_\tau\) 是被控制的宿主生成轨迹，\(s_{1:N}\) 是检测器同步
的水印状态轨迹。主检测器不恢复完整 \(z_\tau\)。

## 2. 原语 P1：密钥域分离

主密钥 \(K\) 至少派生互不混用的子域：

```text
frame_state_transition_key
frame_state_message_key
frame_state_relation_coefficient_key
frame_state_clock_key
frame_state_wrong_key_domain
```

状态转移、PRC/消息、relation coefficient 和同步 pilot/clock 不得直接复用
相同伪随机字节流。
wrong key 必须由预冻结 domain/index 产生，不能从运行结果选择。

公开 Patch relation dictionary 若由 construction 数据得到，其 relation atom
是公开且候选密钥无关的；密钥只控制预声明 relation atom 的系数、符号或受限
低相干组合。Patch 分组、RoPE 轴、attention layer 或输出 Patch 对不得由
candidate key 或 held-out video 决定。

## 3. 原语 P2：帧间水印状态动力学

### 3.1 离散时间

\[
u_{1:N}
=
\operatorname{PRC.Encode}
(K_{\mathrm{message}},\mathrm{context\_digest},M),
\qquad
s_{n+1}=F_Ks_n+G_Ku_n+w_n.
\]

初值和驱动序列必须满足：

\[
s_0=h_{\mathrm{state}}(K,\mathrm{context\_digest}).
\]

PRC 的码族、码长、payload mapping、纠错冗余、交织和解码规则必须在运行前
冻结；candidate key 不得依据待测视频重选 codeword。PRC 负责消息冗余和码序列，
\(F_K,G_K\) 负责可预测动态与缺失状态演化。若 \(F_K\) 被移除后等价于独立逐帧
PRC 解码，则不能声称状态空间贡献。

该公式定义嵌入端：嵌入端持有 \(M\) 并生成 \(u_{1:N}\)。output-only 检测
输入不包含 \(M\)。检测端必须在候选 key 冻结的 PRC 码族、交织、合法消息集合、
消息先验和相同解码复杂度内，对消息假设做离散 marginalization；\(\hat M\)
只能作为检测输出。不得逐帧自由拟合 \(u_n\)，也不得为 owner/wrong key 使用
不同消息搜索空间。

随机状态版本只能使用由独立 calibration 冻结的
\(p_K(s_0\mid\mathrm{context})\)、过程噪声和更新协方差。不得为每个候选 key
自由优化初值、PRC codeword、消息、过程噪声或状态路径。

#### 3.1.1 Public context 原语

`context` 是 output-only 检测端必须同时获得的公开 sidecar/manifest record。
exact key set 冻结为：

```text
context_schema_id =
  "sstw_frame_state_public_context_v1"
method_id =
  "frame_state_synchronized_generative_flow_video_watermark"
protocol_digest =
  64-character lowercase SHA-256 hex
public_nonce_random =
  32-character lowercase hex encoding of a pre-embedding 128-bit nonce
state_clock_rate_numerator =
  positive base-10 integer
state_clock_rate_denominator =
  positive base-10 integer
state_window_count =
  positive base-10 integer
```

禁止额外字段和浮点字段。`public_nonce_random` 必须嵌入前生成并冻结，不得依据
任何生成或检测结果重采样。

canonical serialization 使用 UTF-8 JSON、Unicode code point key sort、
紧凑分隔符 `,`/`:`、无空白和尾随换行、无 Unicode normalization，整数用无
前导零十进制。定义：

\[
\mathrm{context\_digest}
=
\operatorname{SHA256}
\left(\operatorname{canonical\_json}(\mathrm{context})\right).
\]

\(h_{\mathrm{state}}\) 与 PRC 派生只能接收该 digest 作为 context 绑定。嵌入与
检测重算不一致即 fail-closed。Gate 0 检测输入必须准确写为：

```text
video + key + public_context_record
```

不得依赖 prompt、negative prompt、seed、initial noise/latent、内部生成状态、
Flow/velocity trace、carrier trace 或未公开 owner label。不得把可能被重编码
删除的容器 metadata 作为 public context 的唯一载体。

要求：

- \(F_K\) 稳定，不允许状态无界增长；
- 状态不能退化为互不相关逐帧随机码；
- 不同 key 的转移或 PRC 驱动路径必须可分；
- 相同状态集合的不同顺序必须产生不同路径分数；
- 状态能量、PRC 驱动能量和消息恢复必须独立记录；
- raw code bits、PRC/ECC overhead、同步开销、有效 payload 和 fixed-FPR
  payload 必须分开报告。

### 3.2 连续时间

为了处理非整数变速：

\[
\dot s(t_v)=A_Ks(t_v)+G_Ku(t_v),
\]

\[
s(t_v+\Delta t)=\exp(A_K\Delta t)s(t_v)+u_K(\Delta t).
\]

首轮可用离散固定帧率版本；声称连续变速同步前，必须实现并验证
\(\Delta t\)-dependent transition，不能仅对帧序列做重采样。

## 4. 原语 P3：公开 Patch 相对关系 carrier

令视频 DiT token 具有视频窗口索引 \(n\) 与 Patch 索引 \(p\)。公开 relation
dictionary 固定 Patch 分组/配对 \(\mathcal R_j\) 与零和系数
\(a_{j,p}\in\mathbb R^{d_s}\)：

\[
\sum_{p\in\mathcal P_{j,n}}a_{j,p}=0.
\]

状态到 3D-RoPE/相对位置相位的主映射为：

\[
\Delta\theta_{\tau,n,p}
=
\lambda_\theta w(\tau)a_{\rho(n),p}^{\top}c_K(s_n).
\]

若精确实现选择 attention relation bias，则使用：

\[
\Delta b_{\tau,n,p,q}
=
\lambda_b w(\tau)
\left(a_{\rho(n),p}-a_{\rho(n),q}\right)^\top c_K(s_n).
\]

首个 protocol 必须二选一冻结主写入点，不能运行后在 RoPE phase、attention
bias、token residual 或 latent delta 中择优。公开 dictionary 必须精确冻结：

- DiT token layout 与 3D-RoPE axis convention；
- video window 到 latent-token/window 的公开映射 \(\rho\)；
- Patch group/pair、zero-sum coefficient 和 C-order；
- support、overlap、interpolation、boundary handling；
- relation perturbation 的 layer/forward boundary；
- coefficient norm、phase/attention budget 与 sign canonicalization；
- candidate key 到公开 relation atom coefficient 的唯一派生。

不得为候选 key、prompt、seed 或 held-out identity 重选 Patch、layer、RoPE axis
或 relation atom。若输出帧数与 latent/token temporal length 不同，映射必须
在运行前冻结，不能根据可见帧事后移动 support。

直接零和 Patch velocity 控制：

\[
\Delta v_{\tau,n,p}^{\mathrm{direct}}
=
\lambda_v w(\tau)A_pc_K(s_n),
\qquad
\sum_pA_p=0,
\]

只允许作为主 relation carrier 的消融/调试。generic low-frequency latent bank、
单个 decoder-Jacobian atom 或任意 additive direction 不是本原语的等价实现。

## 5. 原语 P4：推理期 Flow 轨迹写入

冻结模型参数下，在同一 base state 和 transformer forward contract 上定义：

\[
\begin{aligned}
v_\tau^{\mathrm{base}}
&=v_\theta(z_\tau,\tau,p;\mathcal R_{\mathrm{base}}),\\
v_\tau^{\mathrm{rel}}
&=v_\theta
(z_\tau,\tau,p;
\mathcal R_{\mathrm{base}}+\Delta\mathcal R_{\tau,n,p}),\\
\Delta v_\tau^{\mathrm{rel}}
&=v_\tau^{\mathrm{rel}}-v_\tau^{\mathrm{base}}.
\end{aligned}
\]

模型参数 \(\theta\) 始终冻结；\(\mathcal R\) 表示本次 forward 的位置/关系坐标，
\(\Delta\mathcal R\) 不是模型参数更新。Euler/Flow state update 为：

\[
z_{\tau+\Delta\tau}
=
z_\tau+\Delta\sigma_\tau
\left(v_\tau^{\mathrm{base}}+
\operatorname{Guard}(\Delta v_\tau^{\mathrm{rel}})\right).
\]

要求：

- \(w(\tau)\) 只描述生成期写入调度；
- 同一条 \(s_{1:N}\) 可在多个 Flow step 被重复强化；
- 不把 Flow step index 编入待检测状态身份；
- 不引入 key-specific Patch/layer/carrier 学习入口；
- clean/zero relation control 必须 exact no-op；
- active relation control 的实际 FP32 velocity delta 非零并通过
  norm/energy/direction guard；
- 记录每个视频窗口、Patch relation 与状态维度的 actual signed exposure；
- 总预算、窗口预算、Patch relation budget 均受控。

生成轨迹机制至少记录 velocity deflection：

\[
\Delta v_{\tau,\perp}^{\mathrm{rel}}
=
P_{v_\tau^{\mathrm{base}}}^{\perp}\Delta v_\tau^{\mathrm{rel}}.
\]

只有冻结的相邻 Flow-step turning/finite-difference statistic 证明了路径弯折，
才允许使用“轨迹曲率编码”表述。Flow Matching/rectified flow 倾向于较直 transport
不等于预训练模型实际路径严格为直线。Flow trace 只用于生成机制审计和
endpoint-only 消融，禁止进入 output-only 主检测输入。

### 5.1 精度路径合同

首个 construction 冻结：

```text
transformer_compute_dtype = bfloat16
scheduler_state_dtype = float32
carrier_control_dtype = float32
actual_delta_measurement_dtype = scheduler_state_dtype
```

强制不变量：

- transformer dtype 与 scheduler-state dtype 分开记录；
- BF16 transformer 输出在控制相加前转换为 FP32 base velocity；
- clean、zero-control、positive、negative 使用相同 scheduler dtype、cast
  次序、控制函数和 state-update 路径；
- zero-control 必须走同一控制函数并得到 exact-zero actual delta，不能绕过；
- actual delta 必须按 scheduler 最终消费 dtype 下的
  `(base + intended_delta) - base` 测量；
- norm、energy、direction、signed exposure 和 waveform symmetry 只使用该
  actual delta；
- 每条记录必须包含 transformer、base velocity、relation control、realized
  relation-induced velocity delta、constrained velocity、scheduler state 和
  actual-delta measurement dtype；
- 任一 branch dtype 漂移、active delta 在 scheduler dtype 下退化为零，或
  zero-control 产生非零 delta，立即 fail-closed。

首轮 construction 只允许一个视频窗口、一个预声明 Patch relation、一个 signed
状态维度，先证明端到端 sign-odd observability。

## 6. 原语 P5：时间保持的 Patch 关系输出特征

从最终保存视频提取：

\[
q_i
=
\phi_{\mathrm{rel}}
(V_{i-w:i+w};\mathcal P_a,\mathcal P_b)
\in\mathbb R^{d_q}.
\]

construction feature 必须：

- 候选密钥无关；
- 在 GPU 运行前冻结；
- 不用 held-out probe 训练；
- 保留局部视频时间索引；
- 使用固定公开 Patch 对/组、频带或通道；
- 使用独立 clean calibration 得到固定中心和尺度；
- 对所有视频只 apply；
- 保存 signed magnitude；
- 不做全视频 latent-time global mean；
- 不做逐视频方向 L2 normalization。

建议最小公开候选族只包含预声明固定编码器 \(\psi\) 上的 Patch 关系：

\[
q_{n,r}
=
\operatorname{Pool}_{p\in\mathcal P_{r,a}}\psi(V_n)_p
-
\operatorname{Pool}_{p\in\mathcal P_{r,b}}\psi(V_n)_p,
\]

以及 \(q_{n+1,r}-q_{n,r}\) 或冻结局部相关/相位。普通 4×4 RGB cell mean 只能
作为 baseline；不能因为其易实现就冒充 Patch-relative feature。

候选族只能在 construction identity \(C_0\) 上按预声明规则选择并冻结；
Gate 0 identity \(A\)、Gate A identity \(B\) 和 Gate B identity \(C\) 均只能
apply，不得重选 Patch pair、encoder layer、频带、窗口或尺度。

## 7. 原语 P6：观测模型

无攻击时：

\[
q_n=T_{\rho(n)}^{\mathrm{rel}}c_K(s_n)+\eta_n,
\qquad
T_j^{\mathrm{rel}}
:=
H_j^{\mathrm{e2e}}\mathcal D_j^{\mathrm{rel}}.
\]

受时间攻击后：

\[
q_i
=
T_{\rho(\pi(i)),\omega_i}^{\mathrm{rel}}
c_K(s_{\pi(i)})
+\eta_i,
\]

其中 \(\omega_i\) 可表示缺失空间块、压缩质量或观测可靠度。
\(H_j^{\mathrm{e2e}}\) 只是从 Patch relation control 到 output relation feature
的概念性未知端到端算子；有限 impulse construction 不能将
\(H_j^{\mathrm{e2e}}\) 与 \(\mathcal D_j^{\mathrm{rel}}\) 分别识别。\(C_0\) 实际
估计和冻结的是受限传递矩阵 \(T_j^{\mathrm{rel}}\)，其列对应公开 relation atom
的端到端响应。不得把 \(H_j^{\mathrm{e2e}}\) 作为独立 construction artifact、
Gate statistic 或 supported claim。

\(\mathcal D_j^{\mathrm{rel}}\)、\(T_j^{\mathrm{rel}}\)、feature 和 whitening
均公开冻结；候选 key 只能改变
\(c_K(\cdot)\)。wrong-key score 不得使用 key-specific feature、传递矩阵或
test-time alignment。Gate 0 只使用无攻击 \(T_j^{\mathrm{rel}}\)；未来
\(T_{j,\omega}^{\mathrm{rel}}\) 必须通过独立 calibration 冻结，不能从单个 attacked test
video 拟合。首轮只能使用 prediction-error/matched-dynamic score，正式联合
协方差未由独立 calibration 估计前不得称 LLR。

受限传递矩阵 \(T_j^{\mathrm{rel}}\) 必须通过实际：

```text
DiT Patch relation control
→ realized Flow velocity deflection
→ generation feedback
→ final latent
→ VAE decode
→ saved video
→ local Patch-relation output feature
```

测量。禁止人为指定满秩 \(H_j^{\mathrm{e2e}}\)，或宣称 impulse probe 单独识别了
\(H_j^{\mathrm{e2e}}\)
之后宣布可观测。

## 8. 原语 P7：时钟跳转模型

对齐路径 \(\pi\) 至少允许以下受限操作：

| 操作 | 路径语义 |
|---|---|
| match | \(\pi(i+1)=\pi(i)+1\) |
| deletion | \(\pi(i+1)>\pi(i)+1\) |
| insertion/repeat | 当前观测不推进或作为 outlier |
| crop | \(\pi(1)\) 未知 |
| speed change | 平均推进率不是1 |

路径代价必须运行前冻结。不能对每个测试视频尝试无限路径后只报告最好结果。
候选速度、最大连续删除、最大插入和起始相位范围必须由 calibration 或协议决定。
\(\pi\) 负责观测索引到水印时钟的对应关系；\(F_K(\Delta t)\) 只负责给定时间间隔
下的状态演化、缺失预测和连续时间状态。实现不得把插帧、裁剪或变速搜索塞进
状态转移矩阵本身。

## 9. 原语 P8：联合状态观测器

对每个候选 key 和合法消息假设 \(M\)，状态先验和驱动必须由检测前冻结的
规则唯一确定：

\[
s_0=h_{\mathrm{state}}(K,\mathrm{context\_digest}),
\qquad
u_{1:N}
=
\operatorname{PRC.Encode}
(K_{\mathrm{message}},\mathrm{context\_digest},M).
\]

若采用随机模型，则 \(p_K(s_0\mid\mathrm{context})\)、过程噪声协方差和更新
协方差必须由独立 calibration 冻结，并对所有 candidate key 使用同一模型族和
复杂度。`context` 不得含从待测输出事后选择的信息。

给定候选 key \(K\)、合法消息假设 \(M\) 和路径 \(\pi\)，observer 执行：

```text
state predict
→ observation predict
→ innovation
→ missing/outlier handling
→ state update
→ path score accumulation
```

预测：

\[
\hat s_{\pi(i)|i-1}
=F_K(\Delta t_i)\hat s_{\pi(i-1)|i-1}.
\]

innovation：

\[
e_i
=
q_i-
T_{\rho(\pi(i))}^{\mathrm{rel}}
c_K(\hat s_{\pi(i)|i-1}).
\]

给定 \(M\) 的非正式首轮分数：

\[
S_{K,M}(\pi)
=-\sum_i e_i^\top W_i e_i-C_{\mathrm{clock}}(\pi).
\]

output-only 的 candidate-key score 必须在同一冻结合法消息集合
\(\mathcal M_{\mathrm{frozen}}\) 与消息先验上 marginalize：

\[
S_K^\star
=
\log\sum_{M\in\mathcal M_{\mathrm{frozen}}}
p_{\mathrm{frozen}}(M)
\exp\left(
\max_{\pi\in\Pi_{\mathrm{frozen}}}S_{K,M}(\pi)
\right).
\]

状态 estimate 只能在上述 key-conditioned prior、过程模型和冻结协方差内更新。
不得把消息搜索空间、\(s_0\)、\(u_n\)、过程噪声、协方差或完整状态路径作为
每个 candidate key 可自由调节的 nuisance parameter。恢复的 \(\hat M\) 是
冻结 PRC posterior 的输出。概率实现应对受约束状态后验积分或使用等价的
规范化滤波分数，不能仅报告每个 key 任意拟合后的最小残差。

若未来使用完整概率模型，必须堆叠相关观测并估计联合协方差；未经独立 calibration
不得把逐帧误差假定为独立并称正式 LLR。

完整 Kalman/filter-smoother、Viterbi/forward DP 或 posterior observer 只有在
Gate C 之后才可进入实现计划；Gate C 之前的 matched-path/order statistic 不能
冒充完整 observer。

## 10. 原语 P9：缺失观测补偿

删除连续帧时，observer 可以执行 \(k\)-step prediction：

\[
\hat s_{n+k|n}=F_K^k\hat s_{n|n}
+\sum_{j=0}^{k-1}F_K^jG_Ku_{n+k-1-j}.
\]

允许置信度下降，但必须满足：

- owner identity 不随机切换成 wrong key；
- 删除区间前后创新恢复；
- 缺失补偿优于独立帧模板；
- 不用原始未攻击视频或隐藏生成状态辅助检测。

## 11. 原语 P10：密钥与顺序证伪

最小机制实验必须包括：

- owner key；
- 预冻结 wrong key；
- correct transition；
- wrong transition；
- reversed transition；
- same-state-set different order；
- shuffled observations；
- static endpoint/whole-video score；
- independent frame/window template；
- edit-distance sequence baseline；
- observer without state update。

只有 correct key、correct order 和正确时钟模型同时占优，才允许称
“状态空间同步”。

## 12. 原语 P11：证据准入与 fixed-FPR 冻结阈值

状态搜索扩大了假设空间，因此 detection decision 必须分为离线 calibration 和
测试时比较两条链路。

离线链路：

```text
predeclared admissibility statistic / rule
→ independent calibration negatives estimate threshold only
→ frozen threshold
```

若 admissibility statistic 或 rule 需要数据拟合，该拟合必须使用独立 development
split；该 split 必须与 fixed-FPR calibration negatives 和 held-out test 隔离。
fixed-FPR calibration negatives 只允许估计 threshold，不能同时用于选择
admissibility rule、feature、\(T_j^{\mathrm{rel}}\)、whitening、clock path cost
或状态先验。

测试链路：

```text
test video + key + public_context_record
→ frozen Patch-relation feature / T_j^rel / clock model
→ runtime score
→ admissibility check
→ compare with frozen threshold
→ held-out decision
```

任何代码、配置、Notebook 或报告不得在每条测试视频上重新估计 threshold、重新选择
admissibility、重新拟合 \(T_j^{\mathrm{rel}}\)、重估 whitening 或更新状态先验。若 calibration
negative 不足，只能报告设计阻断或降级实验，不能把 runtime score 写成 fixed-FPR
检测结果。

## 13. Construction 门禁

### 13.0 Identity 隔离

| Identity | 冻结职责 | 对应门禁 |
|---|---|---|
| \(C_0\) | 构造公开 Patch relation dictionary、feature、受限传递矩阵 \(T_j^{\mathrm{rel}}\) 和 whitening | construction only；不单独识别 \(H_j^{\mathrm{e2e}}\)，不产生 Gate PASS |
| \(A\) | 单窗口×单Patch关系×1维 signed observability，只 apply | Gate 0 |
| \(B\) | 双窗口 Patch 关系可分与组合控制，只 apply | Gate A |
| \(C\) | 跨 identity transfer 与预冻结 wrong-key confirmation，只 apply | Gate B |

\(C_0,A,B,C\) 的 prompt、seed、初始噪声、模型 revision 和用途必须运行前冻结
并写入 governed records。四者不得混用；held-out 数据不得回流重拟合
relation dictionary、feature、\(T_j^{\mathrm{rel}}\)、whitening、threshold 或
状态先验。Gate C 的
identity/data split 必须在 Gate B 通过后另行冻结，当前不授权从上述四者中
事后选择。

### Gate 0：单窗口 Patch 关系 signed observability

在 identity \(A\) 上，一个时间窗口、一个预声明 Patch relation、一个状态方向
的正负控制必须从 DiT relation control 到保存视频关系观测保持：

- 非零 odd response；
- 正负反对称；
- common/even response 受限；
- clean repeat 小于 signed response；
- 实际 exposure 和预算完整。

只有 relation feature 的 primary gate 可产生 Gate 0 PASS。latent scalar、
endpoint projection 或普通 block mean 只能作 checkpoint/baseline。FAIL 立即
停止该 relation carrier/feature，不实现 observer。

### Gate A：双窗口阶段可分

在 identity \(B\) 上，两个视频时间窗口各一个状态方向，要求：

- 每个窗口自身 signed observable；
- cross-window leakage 受限；
- 输出 Patch 关系观测能区分窗口身份；
- 组合控制不被 common-mode 吞没。

注意这里的窗口是视频帧窗口，不是 Flow early/late。

### Gate B：跨 identity 与 key selectivity

在 identity \(C\) 上只 apply，并验证：

- owner transfer 保持；
- wrong key 不匹配；
- 不做 test-time Procrustes、符号翻转或重新 whitening；
- construction identity 不混入 held-out 评分。

### Gate C：短状态路径与顺序

使用一条运行前冻结的短 PRC-driven state path，比较 ordered、reversed、
permuted、wrong transition 和 same-state-set different-order。该 gate 只能使用
运行前预声明的 matched-path/order statistic 验证短路径顺序可辨识。
完整 Kalman/RTS smoother、posterior、时间攻击、fixed-FPR 和 external baseline
均在 Gate C 之后，当前不授权实现。Gate C 通过也不等于 observer 对时间攻击的
优势成立；该优势仍需后续独立 smoke。

## 14. 创新性成立条件

本节只给出可证伪的候选创新成立条件，不是当前 supported claim。当前
Patch-relation carrier、output feature 和 observer 均尚未实现或运行。

论文中只有以下条件同时满足，才能把 Patch 关系、Flow 轨迹写入和状态空间同步
作为联合核心创新：

1. Patch/RoPE relation carrier 优于初始噪声、generic additive velocity 和静态
   output carrier；
2. 实际 Flow velocity deflection/turning 对最终检测有非 endpoint-redundant 作用；
3. PRC-driven 帧状态轨迹可以从最终保存视频 Patch 关系观测恢复；
4. 完整 observer 优于 independent frame template；
5. 完整 observer 优于 edit-distance/bipartite temporal matching；
6. 完整 observer 优于 static whole-video detector；
7. 缺失帧时 prediction/smoothing 提供可测增益；
8. 非整数变速时连续时间转移提供可测增益；
9. wrong key、wrong transition 和 wrong order 均被拒绝；
10. payload 在扣除 PRC/ECC 与同步开销后仍有有效增益，并在 fixed-FPR 下报告。

若只有逐帧消息可解码，而状态转移不产生额外增益，应将方法降级表述为
training-free frame-sequence video watermark，不能声称 state-space contribution。
若只有 additive latent carrier 通过、Patch relation carrier 未实现或无增益，
则不能声称 DiT Patch-relational trajectory watermark。

## 15. 当前实现边界

本文不授权：

- 修改现有 Notebook；
- 启动 GPU/Colab；
- 新增攻击执行；
- 复用旧 Flow-stage Gate A 配置作为新 Gate A；
- 实现完整 observer；
- 在 Gate C 前实现 posterior、完整 smoother、时间攻击或 fixed-FPR；
- 从单个测试视频拟合 \(T_{j,\omega}^{\mathrm{rel}}\)、可靠度模型、threshold 或
  whitening；
- 更新 Drive request；
- 生成正式结果或推进项目阶段。

当前真实 frame-state Gate 0 的 decoder-Jacobian additive atom 已在 decoded/
saved-video signed response 与 held-out transfer 上 FAIL。它证明该 surrogate
组合应停止，不证明 Patch relation carrier、PRC-driven state path 或 observer
失败。当前代码尚未实现本文件的 3D-RoPE/relative-attention carrier 和对应
Patch-relation output feature。

若未来获得独立实现授权，首个任务才可单独冻结 Gate 0 的单一 Patch relation
carrier、对应 relation feature、identity 分离和最小视频计划；本合同本身不
授权实现，也不授权在代码审核后自动运行 Colab。
generic public low-frequency bank 只可作为 baseline/fallback，不得替代该主方法
Gate 后仍声称 SSTW 已实现。
