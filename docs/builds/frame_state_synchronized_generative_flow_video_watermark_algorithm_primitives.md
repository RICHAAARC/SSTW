# 状态空间同步的生成 Flow 视频水印算法原语

## 0. 适用范围

本文只冻结方法语义和可证伪原语，不冻结尚未通过 construction 的具体数值、
threshold、状态维度或攻击网格。当前：

```text
formal_result=false
stage_progression_allowed=false
runtime_implementation_authorized=false
```

历史 Flow-stage carrier、prompt-conditioned replay 和全局 VAE feature 均不得作为
本算法原语已成立的证据。

## 1. 符号与双时间坐标

| 符号 | 含义 |
|---|---|
| \(\tau\) | 视频生成模型的 Flow/sampler 时间 |
| \(n\) | 原始水印视频帧或固定时间窗口索引 |
| \(i\) | 受攻击视频中的观测帧/窗口索引 |
| \(z_\tau\) | 视频模型高维生成 latent |
| \(s_n\) | 低维帧间水印状态 |
| \(m_n\) | 密钥派生的水印驱动/消息 |
| \(\pi(i)\) | 观测索引到水印时钟的对齐路径 |
| \(q_i\) | 从最终视频提取的局部观测 |

强制不变量：

```text
watermark_state_clock = video_frame_time
embedding_schedule_clock = generative_flow_time
```

任何代码不得使用同一个未区分的 `t` 同时表示两条时间轴。

## 2. 原语 P1：密钥域分离

主密钥 \(K\) 至少派生互不混用的子域：

```text
frame_state_transition_key
frame_state_message_key
frame_state_carrier_key
frame_state_clock_key
frame_state_wrong_key_domain
```

状态转移、消息、carrier 和同步 pilot/clock 不得直接复用相同伪随机字节流。
wrong key 必须由预冻结 domain/index 产生，不能从运行结果选择。

公开 output carrier dictionary 若由 construction 数据得到，其 atom 是公开且
候选密钥无关的；密钥只控制预声明 atom 的符号、旋转或低相干组合。

## 3. 原语 P2：帧间水印状态动力学

### 3.1 离散时间

\[
s_{n+1}=F_Ks_n+G_Km_n+w_n.
\]

初值和驱动序列必须满足：

\[
s_0=h_{\mathrm{state}}(K,\mathrm{context}),
\qquad
m_n=h_{\mathrm{message}}(K,\mathrm{context},n).
\]

随机版本只能使用由独立 calibration 冻结的
\(p_K(s_0\mid\mathrm{context})\)、过程噪声和更新协方差。不得为每个候选 key
自由优化初值、消息、过程噪声或状态路径。

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

\(h_{\mathrm{state}}\) 与 \(h_{\mathrm{message}}\) 只能接收该 digest 作为
context 绑定。嵌入与检测重算不一致即 fail-closed。Gate 0 检测输入必须准确写为：

```text
video + key + public_context_record
```

不得依赖 prompt、negative prompt、seed、initial noise/latent、内部生成状态、
Flow/velocity trace、carrier trace 或未公开 owner label。不得把可能被重编码
删除的容器 metadata 作为 public context 的唯一载体。

要求：

- \(F_K\) 稳定，不允许状态无界增长；
- 状态不能退化为互不相关逐帧随机码；
- 不同 key 的转移或驱动路径必须可分；
- 相同状态集合的不同顺序必须产生不同路径分数；
- 状态能量和消息能量必须独立记录。

### 3.2 连续时间

为了处理非整数变速：

\[
\dot s(t_v)=A_Ks(t_v)+G_Km(t_v),
\]

\[
s(t_v+\Delta t)=\exp(A_K\Delta t)s(t_v)+u_K(\Delta t).
\]

首轮可用离散固定帧率版本；声称连续变速同步前，必须实现并验证
\(\Delta t\)-dependent transition，不能仅对帧序列做重采样。

## 4. 原语 P3：帧状态控制场

令视频 latent layout 为 `[B,C,T,H,W]`。帧/窗口状态控制场为：

\[
C_K(s_{1:N})[:,:,n,:,:]
=D_{\rho(n)}c_K(s_n).
\]

\(D_j\) 是由 construction identity 构造并公开冻结的候选密钥无关 dictionary；
\(c_K(s_n)\) 只负责由 key 和状态确定公开 atom 的系数、符号、旋转或低相干
组合。不得为候选 key 或 held-out identity 重新学习 dictionary。若代码保留
\(B_{K,n}\)，必须满足
\(B_{K,n}s_n\equiv D_{\rho(n)}c_K(s_n)\)，且不能形成第二套隐式 carrier。

若 latent temporal length 与输出帧数不同，必须预声明公开映射：

\[
n_{\mathrm{latent}}=\rho(n_{\mathrm{video}}),
\]

并冻结：

- support window；
- overlap；
- interpolation；
- boundary handling；
- 每个输出帧对 latent slice 的责任；
- carrier temporal coherence。

不得根据生成结果事后移动 support 或选择可见帧。

## 5. 原语 P4：推理期 Flow 注入

冻结模型参数下：

\[
\Delta v_{\tau,n}
=
\lambda w(\tau)D_{\rho(n)}c_K(s_n),
\qquad
C_K(s_{1:N})[:,:,n,:,:]
=D_{\rho(n)}c_K(s_n).
\]

若写成完整速度场，只能作为：

\[
v_\tau^{\mathrm{wm}}
=v_\theta(z_\tau,\tau,p)
+\lambda w(\tau)C_K(s_{1:N}).
\]

Euler/Flow update 为：

\[
z_{\tau+\Delta\tau}
=z_\tau+\Delta\sigma_\tau v_\tau^{\mathrm{wm}}.
\]

要求：

- \(w(\tau)\) 只描述生成期写入调度；
- 同一条 \(s_{1:N}\) 可在多个 Flow step 被重复强化；
- 不把 Flow step index 编入待检测状态身份；
- 不引入任何 key-specific carrier 学习入口；\(B_{K,n}\) 若出现只能是
  \(D_{\rho(n)}c_K(s_n)\) 的派生别名；
- clean 分支 exact no-op；
- active 分支实际 FP32 delta 非零且通过 norm/energy/direction guard；
- 记录每个视频时间 support 的 actual signed exposure；
- 总预算与局部帧/窗口预算都必须受控。

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
- 每条记录必须包含 transformer、base velocity、carrier、constrained velocity、
  scheduler state 和 actual-delta measurement dtype；
- 任一 branch dtype 漂移、active delta 在 scheduler dtype 下退化为零，或
  zero-control 产生非零 delta，立即 fail-closed。

首轮 construction 只允许一个帧/窗口、一个 signed 状态维度，先证明端到端
sign-odd observability。

## 6. 原语 P5：时间保持的输出特征

从最终保存视频提取：

\[
q_i=\phi_{\mathrm{local}}(V_{i-w:i+w})\in\mathbb R^{d_q}.
\]

construction feature 必须：

- 候选密钥无关；
- 在 GPU 运行前冻结；
- 不用 held-out probe 训练；
- 保留局部视频时间索引；
- 使用固定空间块/频带/通道；
- 使用独立 clean calibration 得到固定中心和尺度；
- 对所有视频只 apply；
- 保存 signed magnitude；
- 不做全视频 latent-time global mean；
- 不做逐视频方向 L2 normalization。

建议最小公开候选族只包含预声明的：

\[
\bar h_n-\mu_{\mathrm{clean}},
\qquad
\bar h_{n+1}-\bar h_n.
\]

候选族只能在 construction identity \(C_0\) 上选择并冻结；Gate 0 identity
\(A\)、Gate A identity \(B\) 和 Gate B identity \(C\) 均只能 apply，不得
重新选择。

## 7. 原语 P6：观测模型

无攻击时：

\[
q_n=T_{\rho(n)}c_K(s_n)+\eta_n,
\qquad
T_j:=H_jD_j.
\]

受时间攻击后：

\[
q_i
=
T_{\rho(\pi(i)),\omega_i}
c_K(s_{\pi(i)})
+\eta_i,
\]

其中 \(\omega_i\) 可表示缺失空间块、压缩质量或观测可靠度。\(H_j\) 只是从
latent carrier field 到 output feature 的概念性未知端到端算子；有限 impulse
construction 不能将 \(H_j\) 与 \(D_j\) 分别识别。\(C_0\) 实际估计和冻结的是
受限传递矩阵 \(T_j=H_jD_j\)，其列对应公开 dictionary atom 的端到端响应。
不得把 \(H_j\) 作为独立 construction artifact、Gate statistic 或 supported
claim。

\(D_j\)、\(T_j\)、feature 和 whitening 均公开冻结；候选 key 只能改变
\(c_K(\cdot)\)。wrong-key score 不得使用 key-specific feature、传递矩阵或
test-time alignment。Gate 0 只使用无攻击 \(T_j\)；未来
\(T_{j,\omega}\) 必须通过独立 calibration 冻结，不能从单个 attacked test
video 拟合。首轮只能使用 prediction-error/matched-dynamic score，正式联合
协方差未由独立 calibration 估计前不得称 LLR。

受限传递矩阵 \(T_j\) 必须通过实际：

```text
Flow control
→ generation feedback
→ final latent
→ VAE decode
→ saved video
→ local output feature
```

测量。禁止人为指定满秩 \(H_j\)，或宣称 impulse probe 单独识别了 \(H_j\)
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

每个候选 key 的状态先验和驱动必须在检测前固定：

\[
s_0=h_{\mathrm{state}}(K,\mathrm{context}),
\qquad
m_n=h_{\mathrm{message}}(K,\mathrm{context},n).
\]

若采用随机模型，则 \(p_K(s_0\mid\mathrm{context})\)、过程噪声协方差和更新
协方差必须由独立 calibration 冻结，并对所有 candidate key 使用同一模型族和
复杂度。`context` 不得含从待测输出事后选择的信息。

给定候选 key \(K\) 和路径 \(\pi\)，observer 执行：

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
T_{\rho(\pi(i))}
c_K(\hat s_{\pi(i)|i-1}).
\]

非正式首轮分数：

\[
S_K(\pi)
=-\sum_i e_i^\top W_i e_i-C_{\mathrm{clock}}(\pi).
\]

最终检测：

\[
S_K^\star=\max_{\pi\in\Pi_{\mathrm{frozen}}}S_K(\pi).
\]

状态 estimate 只能在上述 key-conditioned prior、过程模型和冻结协方差内更新。
不得把 \(s_0\)、\(m_n\)、过程噪声、协方差或完整状态路径作为每个 candidate
key 可自由调节的 nuisance parameter。概率实现应对受约束状态后验积分或使用
等价的规范化滤波分数，不能仅报告每个 key 任意拟合后的最小残差。

若未来使用完整概率模型，必须堆叠相关观测并估计联合协方差；未经独立 calibration
不得把逐帧误差假定为独立并称正式 LLR。

完整 Kalman/filter-smoother、Viterbi/forward DP 或 posterior observer 只有在
Gate C 之后才可进入实现计划；Gate C 之前的 matched-path/order statistic 不能
冒充完整 observer。

## 10. 原语 P9：缺失观测补偿

删除连续帧时，observer 可以执行 \(k\)-step prediction：

\[
\hat s_{n+k|n}=F_K^k\hat s_{n|n}
+\sum_{j=0}^{k-1}F_K^jG_Km_{n+k-1-j}.
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
admissibility rule、feature、\(T_j\)、whitening、clock path cost 或状态先验。

测试链路：

```text
test video + key + public_context_record
→ frozen feature / T_j / clock model
→ runtime score
→ admissibility check
→ compare with frozen threshold
→ held-out decision
```

任何代码、配置、Notebook 或报告不得在每条测试视频上重新估计 threshold、重新选择
admissibility、重新拟合 \(T_j\)、重估 whitening 或更新状态先验。若 calibration
negative 不足，只能报告设计阻断或降级实验，不能把 runtime score 写成 fixed-FPR
检测结果。

## 13. Construction 门禁

### 13.0 Identity 隔离

| Identity | 冻结职责 | 对应门禁 |
|---|---|---|
| \(C_0\) | 构造公开 dictionary、feature、受限传递矩阵 \(T_j=H_jD_j\) 和 whitening | construction only；不单独识别 \(H_j\)，不产生 Gate PASS |
| \(A\) | 单窗口×1维 signed observability，只 apply | Gate 0 |
| \(B\) | 双窗口可分与组合控制，只 apply | Gate A |
| \(C\) | 跨 identity transfer 与预冻结 wrong-key confirmation，只 apply | Gate B |

\(C_0,A,B,C\) 的 prompt、seed、初始噪声、模型 revision 和用途必须运行前冻结
并写入 governed records。四者不得混用；held-out 数据不得回流重拟合
dictionary、feature、\(T_j\)、whitening、threshold 或状态先验。Gate C 的
identity/data split 必须在 Gate B 通过后另行冻结，当前不授权从上述四者中
事后选择。

### Gate 0：单窗口 signed observability

在 identity \(A\) 上，一个时间窗口、一个状态方向的正负控制必须从 latent
到保存视频保持：

- 非零 odd response；
- 正负反对称；
- common/even response 受限；
- clean repeat 小于 signed response；
- 实际 exposure 和预算完整。

FAIL 立即停止该 carrier/feature，不实现 observer。

### Gate A：双窗口阶段可分

在 identity \(B\) 上，两个视频时间窗口各一个状态方向，要求：

- 每个窗口自身 signed observable；
- cross-window leakage 受限；
- 输出观测能区分窗口身份；
- 组合控制不被 common-mode 吞没。

注意这里的窗口是视频帧窗口，不是 Flow early/late。

### Gate B：跨 identity 与 key selectivity

在 identity \(C\) 上只 apply，并验证：

- owner transfer 保持；
- wrong key 不匹配；
- 不做 test-time Procrustes、符号翻转或重新 whitening；
- construction identity 不混入 held-out 评分。

### Gate C：短状态路径与顺序

比较 ordered、reversed、permuted 和 same-state-set different-order。该 gate
只能使用运行前预声明的 matched-path/order statistic 验证短路径顺序可辨识。
完整 Kalman/RTS smoother、posterior、时间攻击、fixed-FPR 和 external baseline
均在 Gate C 之后，当前不授权实现。Gate C 通过也不等于 observer 对时间攻击的
优势成立；该优势仍需后续独立 smoke。

## 14. 创新性成立条件

论文中只有以下条件同时满足，才能把状态空间作为核心创新：

1. Flow嵌入比初始噪声/静态输出载体具有明确作用；
2. 帧状态轨迹可以从最终保存视频局部观测；
3. 完整 observer 优于 independent frame template；
4. 完整 observer 优于 edit-distance/bipartite temporal matching；
5. 完整 observer 优于 static whole-video detector；
6. 缺失帧时 prediction/smoothing 提供可测增益；
7. 非整数变速时连续时间转移提供可测增益；
8. wrong key、wrong transition 和 wrong order 均被拒绝。

若只有逐帧消息可解码，而状态转移不产生额外增益，应将方法降级表述为
training-free frame-sequence video watermark，不能声称 state-space contribution。

## 15. 当前实现边界

本文不授权：

- 修改现有 Notebook；
- 启动 GPU/Colab；
- 新增攻击执行；
- 复用旧 Flow-stage Gate A 配置作为新 Gate A；
- 实现完整 observer；
- 在 Gate C 前实现 posterior、完整 smoother、时间攻击或 fixed-FPR；
- 从单个测试视频拟合 \(T_{j,\omega}\)、可靠度模型、threshold 或 whitening；
- 更新 Drive request；
- 生成正式结果或推进项目阶段。

下一实现任务应先单独冻结 Gate 0 的 carrier、local feature、identity 分离和最小
视频计划，再经独立代码审核后决定是否运行 Colab。
